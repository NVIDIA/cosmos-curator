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

"""Automatic stage instrumentation and profiling backends.

This module wraps pipeline stages with pluggable profiling backends so
that every ``process_data()``, ``stage_setup()``, and
``stage_setup_on_node()`` call is automatically instrumented without
requiring any changes to stage code.

Current backends:

* **CPU** -- pyinstrument flame-trees (``--profile-cpu``).
* **Memory** -- memray heap captures (``--profile-memory``).
* **GPU** -- torch.profiler CUDA traces (``--profile-gpu``).
* **Tracing** -- OpenTelemetry distributed spans via Ray (``--profile-tracing``).

Adding a new per-stage backend requires only local changes inside
``_ProfilingState`` -- no stage code or pipeline wiring changes.

The tracing backend is different: it operates at the Ray cluster level
(not per-stage) via ``_tracing_startup_hook``, configured in
``profiling_scope()`` before the pipeline starts.

Public API:

``profiling_wrapper(stage, config)``
    Swap a stage's class with a dynamic subclass that instruments
    every ``process_data()`` call.

``profiling_scope(args, *, stage_name, label)``
    Context manager for root-level / driver-process profiling.
    Also sets up Ray distributed tracing when ``--profile-tracing``
    is enabled.

Internal helpers:

``_apply_profiling_config(args)``
    Derive a ``ProfilingConfig`` from parsed CLI arguments and
    set ``args.perf_profile = True`` when any backend is enabled.
    Auto-discovers the pipeline output path and appends ``/profile``.
"""

import argparse
import contextlib
import os
import pathlib
import tempfile
import time
from collections.abc import Generator
from typing import Any

import attrs
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, PipelineTask
from cosmos_curator.core.utils.artifacts.delivery import ArtifactDelivery
from cosmos_curator.core.utils.infra.tracing import (
    TracedSpan,
    artifact_id,
    process_tag,
    trace_root_anchor,
    traced_span,
)


@attrs.define(frozen=True)
class ProfilingConfig:
    """Frozen configuration governing which profiling backends are active.

    Every pipeline stage goes through a well-defined lifecycle inside
    its Ray actor.  Profiling must capture all phases -- not just the
    hot ``process_data()`` loop -- because setup can dominate wall-clock
    time (e.g. model weight copies, GPU context init) and is invisible
    to per-task metrics::

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
        destroy()                 <-- flush final artifacts

    ``profiling_wrapper()`` creates a dynamic subclass of each stage
    that intercepts every lifecycle method and delegates to
    ``_ProfilingState``, which orchestrates the per-stage backends:

    * **CPU** -- pyinstrument (``--profile-cpu``).  Produces per-call
      HTML flame-trees and ``.pyisession`` files.
    * **Memory** -- memray (``--profile-memory``).  Produces per-call
      ``.bin`` heap captures and HTML flamegraphs.
    * **GPU** -- torch.profiler (``--profile-gpu``).  Produces per-call
      Chrome Trace JSON files viewable in Perfetto.
    * **Tracing** -- OpenTelemetry via Ray (``--profile-tracing``).
      Operates at the cluster level, not per-stage.  Configured in
      ``profiling_scope()`` before the pipeline starts.

    Backend start/stop follows LIFO nesting to unwind
    ``sys.setprofile`` hooks correctly::

        start :  pyinstrument --> memray --> torch.profiler
        stop  :  torch.profiler --> memray --> pyinstrument

    Each backend derives its output subdirectory from the local
    staging directory (e.g. ``staging_dir/cpu/``, ``staging_dir/memory/``).
    During the pipeline run, all artifacts are written to a local
    staging directory via ``pathlib.Path``.  Post-pipeline,
    ``ArtifactDelivery`` collects them from all nodes via
    ``RayFileTransport`` and uploads to the final
    ``<output-path>/profile`` directory.

    Artifact files follow a single naming convention via
    ``artifact_id()``::

        <StageName>_<label>_<count>_<hostname>_<pid>.<ext>

    Examples:
          MyStage_setup_on_node_1_node03_5819.html       (cpu)
          MyStage_process_data_3_node03_10552.bin         (memory)
          MyStage_process_data_1_gpu01_42.json            (gpu)

    The same ``profiling.artifact_id`` string is attached to the OTel
    span so traces can be cross-referenced with profiling files.

    Per-backend exclude lists allow disabling individual backends for
    specific scopes.  The ``_root`` driver scope is typically excluded
    from memory/GPU profiling because its lifetime is dominated by
    idle orchestration time.

    Attributes:
        profile_dir: Base output directory for all profiling backends.
            Always the static subdirectory ``profile/`` under the
            pipeline output path (e.g. ``<output_clip_path>/profile``)
            for simplicity -- no configurable override exists.
            Individual backends derive subdirectories (``cpu/``,
            ``memory/``, ``gpu/``, ``traces/``) from this path.
        s3_profile_name: Named credential profile for S3/Azure storage
            access when uploading profiling artifacts.  Corresponds
            to the ``--output-s3-profile-name`` CLI flag.
        cpu_enabled: Enable CPU flame-tree profiling (pyinstrument).
        memory_enabled: Enable heap memory allocation profiling (memray).
        gpu_enabled: Enable GPU kernel/operator profiling (torch.profiler).
        tracing_enabled: Enable distributed OpenTelemetry tracing via
            Ray's ``_tracing_startup_hook``.  Captures cross-actor
            spans (scheduling, method invocations) as NDJSON files.
        tracing_sampling: Trace sampling rate (0.0--1.0) passed to
            ``enable_tracing()`` which sets standard OTel env vars.
        tracing_otlp_endpoint: OTLP HTTP collector endpoint for
            remote span export.  Empty string (default) disables
            OTLP -- only the local file exporter is active.  Set via
            ``--profile-tracing-otlp-endpoint`` or the standard
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
        staging_dir: Base staging directory for profiling and trace
            artifacts.  Empty string means "use env var or create a
            temp dir".  Populated by ``_apply_profiling_config()``
            from ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR`` (set by
            ``ArtifactDelivery.create()``), so the path is serialized
            with the stage config and available on workers even when
            the env var is not inherited (e.g. ``ray job submit``
            in NVCF).
        traceparent: Driver's trace anchor context in
            ``"{trace_id_hex}:{span_id_hex}"`` format.  May be empty
            on the snapshot taken in ``_apply_profiling_config()`` if
            that runs before ``propagate_trace_context()`` (common when
            stage specs are built early).  Workers fall back to
            ``COSMOS_CURATOR_TRACEPARENT`` in the environment at span
            start.  When both are set, they should match.
        cpu_exclude: Scope names to exclude from CPU profiling.
            Matches stage class names (e.g. ``"MyStage"``)
            and the special ``"_root"`` driver scope.
        memory_exclude: Scope names to exclude from memory profiling.
        gpu_exclude: Scope names to exclude from GPU profiling.

    """

    profile_dir: str = "./profile"
    s3_profile_name: str = "default"
    cpu_enabled: bool = False
    memory_enabled: bool = False
    gpu_enabled: bool = False
    tracing_enabled: bool = False
    tracing_sampling: float = 0.01
    tracing_otlp_endpoint: str = ""
    staging_dir: str = ""
    traceparent: str = ""
    cpu_exclude: frozenset[str] = frozenset()
    memory_exclude: frozenset[str] = frozenset()
    gpu_exclude: frozenset[str] = frozenset()


# The instrumentation wrapper creates a dynamic subclass of the stage
# class (same pattern as ``stage_save_wrapper`` in ``stage_replay.py``)
# that wraps ``process_data()`` with ``_ProfilingState.start()`` /
# ``_ProfilingState.stop_and_save()``.
#
# Currently supported backends:
#   - pyinstrument  (CPU flame-tree profiling)
#   - memray        (heap memory profiling)
#
# The design is extensible -- adding a new backend (e.g. custom
# metrics) only requires changes inside
# ``_ProfilingState``; no stage code or pipeline wiring changes.
#
# Key properties:
#   * Requires zero changes to 30+ stage constructors.
#   * Works transparently with all executors (Xenna, StageRunner,
#     SequentialRunner) because instrumentation lives inside
#     ``process_data()``.
#   * Keeps ``StageTimer`` focused on per-task stats.
#
#                       +------------------------------+
#                       |      CuratorStage (base)     |
#                       |  process_data(tasks) -> ...  |
#                       +------------------------------+
#                                     ^
#                                     | (dynamic subclass)
#                       +------------------------------+
#                       |    _ProfiledStage(base)      |
#                       |  process_data(tasks):        |
#                       |    state.start()             |
#                       |    super().process_data()    |
#                       |    state.stop_and_save()     |
#                       +------------------------------+


def _resolve_staging_path(staging_dir: pathlib.Path, sub_path: str) -> pathlib.Path:
    """Resolve *sub_path* under *staging_dir*, ensure parents exist, remove stale file.

    All profiling backends call this before writing an artifact so
    that the staging directory structure mirrors the final output
    layout (``cpu/``, ``memory/``, ``gpu/``, ``traces/``).

    Removing any pre-existing file guarantees a fresh path -- tools
    that refuse to overwrite (e.g. memray ``Tracker``) work without
    pre-cleanup from the caller.

    Args:
        staging_dir: Root staging directory on this node.
        sub_path: Relative path within the staging directory
            (e.g. ``"cpu/MyStage_setup_1_node03_5819.html"``).

    Returns:
        Absolute path to the resolved file location.

    """
    dest = staging_dir / sub_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    return dest


class _CpuProfilingBackend:
    """CPU profiling backend using pyinstrument.

    Manages a single pyinstrument ``Profiler`` lifecycle per
    ``process_data()`` call, producing per-call HTML flame-tree
    reports and ``.pyisession`` files.

    All methods are safe to call on retries: leftover state from a
    failed previous attempt is cleaned up before starting a new
    session.  Runtime errors disable the backend for the actor
    lifetime instead of crashing the pipeline.

    Attributes:
        _profiler: Active per-call pyinstrument ``Profiler`` or ``None``.
        _cpu_disabled: ``True`` after an unrecoverable runtime
            error is encountered.

    """

    def __init__(
        self,
        stage_name: str,
        staging_dir: pathlib.Path,
        *,
        enabled: bool = False,
        exclude: frozenset[str] = frozenset(),
    ) -> None:
        self._stage_name = stage_name
        self._staging_dir = staging_dir
        self._profiler: Any = None
        self._cpu_disabled = not enabled or stage_name in exclude
        self._start_time: float = 0.0
        self._call_id: str = ""
        self._id = f"{stage_name}_{process_tag()}"

        if self._cpu_disabled and enabled:
            logger.debug(f"[pyinstrument] {self._id}: excluded by --profile-cpu-exclude")

    def start(self, call_id: str) -> None:
        """Start a new pyinstrument session.

        Safe to call on retries: if a profiler is already running it
        is stopped first so the new session starts cleanly.  Errors
        from pyinstrument disable CPU profiling for the rest of the
        actor lifetime rather than crashing the pipeline.

        Args:
            call_id: Human-readable call identifier for file naming
                (e.g. ``"setup_on_node_1"``, ``"process_data_3"``).

        """
        if self._cpu_disabled:
            return
        self._call_id = call_id

        # Guard against leftover profiler from a failed previous attempt.
        if self._profiler is not None:
            self.stop_on_error()

        try:
            from pyinstrument import Profiler  # noqa: PLC0415

            self._start_time = time.perf_counter()
            self._profiler = Profiler()
            self._profiler.start()
        except Exception as e:  # noqa: BLE001  -- profiling must never crash the pipeline
            logger.warning(
                f"[pyinstrument] {self._id}: failed to start; CPU profiling disabled for this actor: {e}",
                exc_info=True,
            )
            self.stop_on_error()
            self._cpu_disabled = True

    def stop_on_error(self) -> None:
        """Stop pyinstrument without saving artifacts (error rollback)."""
        if self._profiler is None:
            return
        with contextlib.suppress(RuntimeError):
            self._profiler.stop()
        self._profiler = None

    def stop(self) -> None:
        """Stop the profiler, dump text to stdout, and save HTML + session."""
        if self._profiler is None:
            return
        self._profiler.stop()
        elapsed = time.perf_counter() - self._start_time
        session = self._profiler.last_session

        # Dump concise text to stdout for immediate visibility.
        # Wall-clock elapsed time is included so the user can see the
        # real duration without having to open the HTML report.
        logger.info(
            f"[pyinstrument] {self._id} call #{self._call_id} "
            f"(wall {elapsed:.3f}s)\n"
            f"{self._profiler.output_text(unicode=False, color=False)}",
        )

        # Save per-call HTML and .pyisession to the local staging dir.
        # File names include hostname and PID so that artifacts from
        # different actors / nodes never collide when collected.
        base = artifact_id(self._stage_name, self._call_id)
        html_sub = f"cpu/{base}.html"
        _resolve_staging_path(self._staging_dir, html_sub).write_text(
            self._profiler.output_html(),
            encoding="utf-8",
        )

        session_sub = f"cpu/{base}.pyisession"
        path = _resolve_staging_path(self._staging_dir, session_sub)
        if session is not None:
            session.save(path)

        logger.debug(f"[pyinstrument] {self._id}: saved: {self._staging_dir}/{html_sub}")
        self._profiler = None

    def flush(self) -> None:
        """No-op -- all CPU artifacts are flushed per-call in ``stop()``."""


class _MemoryProfilingBackend:
    """Memory profiling backend using memray.

    Manages a ``memray.Tracker`` context manager per
    ``process_data()`` call, producing ``.bin`` captures and
    HTML flamegraph reports.

    All methods are safe to call on retries: leftover state from a
    failed previous attempt is cleaned up before starting a new
    tracker.  Runtime errors disable the backend for the actor
    lifetime instead of crashing the pipeline.

    Attributes:
        _memray_tracker: Active memray ``Tracker`` context or ``None``.
        _bin_path: :class:`WritablePath` to the current ``.bin``
            capture (resolved by :meth:`StorageWriter.resolve_path`).
        _mem_disabled: ``True`` after an unrecoverable runtime
            error is encountered.

    """

    def __init__(
        self,
        stage_name: str,
        staging_dir: pathlib.Path,
        *,
        enabled: bool = False,
        exclude: frozenset[str] = frozenset(),
    ) -> None:
        self._stage_name = stage_name
        self._staging_dir = staging_dir
        self._memray_tracker: Any = None
        self._bin_path: pathlib.Path | None = None
        self._mem_disabled = not enabled or stage_name in exclude
        self._call_id: str = ""
        self._id = f"{stage_name}_{process_tag()}"

        if self._mem_disabled and enabled:
            logger.debug(f"[memray] {self._id}: excluded by --profile-memory-exclude")

    def stop_on_error(self) -> None:
        """Exit memray tracker without generating reports (error rollback).

        The staging file is left in place for debugging; no upload is
        attempted.
        """
        if self._memray_tracker is None:
            return
        with contextlib.suppress(Exception):
            self._memray_tracker.__exit__(None, None, None)
        self._memray_tracker = None
        self._bin_path = None

    def start(self, call_id: str) -> None:
        """Start a memray tracker writing to a .bin file.

        Safe to call on retries: cleans up leftover tracker state
        from a failed previous attempt.  Errors from memray disable
        memory profiling for the rest of the actor lifetime rather
        than crashing the pipeline.

        Args:
            call_id: Human-readable call identifier for file naming
                (e.g. ``"setup_on_node_1"``, ``"process_data_3"``).

        """
        if self._mem_disabled:
            return
        self._call_id = call_id

        # Guard against leftover tracker from a failed previous attempt.
        if self._memray_tracker is not None:
            self.stop_on_error()

        # Resolve a writable local path under the staging directory.
        base = artifact_id(self._stage_name, self._call_id)
        self._bin_path = _resolve_staging_path(self._staging_dir, f"memory/{base}.bin")

        # ``native_traces``  -- captures C-extension stacks; worthwhile
        #     for GPU/CUDA workloads.
        # ``trace_python_allocators`` -- surfaces pymalloc arena
        #     subdivisions as independent allocations for finer
        #     Python-level granularity.
        try:
            from memray import Tracker  # noqa: PLC0415

            self._memray_tracker = Tracker(
                file_name=str(self._bin_path),
                native_traces=True,
                trace_python_allocators=True,
                follow_fork=True,
            )
            self._memray_tracker.__enter__()
        except Exception as e:  # noqa: BLE001  -- profiling must never crash the pipeline
            logger.warning(
                f"[memray] {self._id}: failed to start; memory profiling disabled for this actor: {e}",
                exc_info=True,
            )
            self.stop_on_error()
            self._mem_disabled = True

    def stop(self) -> None:
        """Stop the tracker, dump stats to stdout, and generate an HTML flamegraph.

        The ``Tracker.__exit__()`` call may fail when memray and
        pyinstrument are both active because memray tries to restore
        pyinstrument's ``sys.setprofile`` hook (a C-level
        ``ProfilerState`` object) by calling it, which raises
        ``TypeError``.  The ``.bin`` capture file is flushed
        continuously by memray, so it still contains valid data up to
        the point of failure.  We therefore catch the ``__exit__``
        error and continue with stats / flamegraph generation.
        """
        if self._memray_tracker is None:
            return

        # Stop the tracker.  __exit__() can fail when memray conflicts
        # with pyinstrument's sys.setprofile hook.  The .bin file is
        # still usable -- memray flushes data continuously during the
        # profiling session.
        with contextlib.suppress(Exception):
            self._memray_tracker.__exit__(None, None, None)
        self._memray_tracker = None

        if self._bin_path is None or not self._bin_path.exists():
            return

        # Dump concise stats to stdout for immediate visibility.
        try:
            from memray._memray import compute_statistics  # noqa: PLC0415
            from memray.reporters.stats import StatsReporter  # noqa: PLC0415

            stats = compute_statistics(
                str(self._bin_path),
                report_progress=False,
                num_largest=5,
            )
            logger.info(
                f"[memray] {self._id} call #{self._call_id}",
            )
            StatsReporter(stats, num_largest=5).render()
        except Exception as e:  # noqa: BLE001  -- profiling must never crash the pipeline
            logger.warning(
                f"[memray] {self._id}: failed to generate stats for {self._bin_path}: {e}",
                exc_info=True,
            )

        # Generate HTML flamegraph to the local staging directory.
        base = artifact_id(self._stage_name, self._call_id)
        html_sub = f"memory/{base}.html"
        try:
            from memray import FileReader  # noqa: PLC0415
            from memray.reporters.flamegraph import FlameGraphReporter  # noqa: PLC0415

            reader = FileReader(str(self._bin_path))
            snapshot = reader.get_high_watermark_allocation_records(
                merge_threads=True,
            )
            reporter = FlameGraphReporter.from_snapshot(
                snapshot,
                memory_records=tuple(reader.get_memory_snapshots()),
                native_traces=reader.metadata.has_native_traces,
            )
            html_path = _resolve_staging_path(self._staging_dir, html_sub)
            with html_path.open("w", encoding="utf-8") as buf:
                reporter.render(
                    outfile=buf,
                    metadata=reader.metadata,
                    show_memory_leaks=False,
                    merge_threads=True,
                    inverted=False,
                )
            logger.debug(f"[memray] {self._id}: flamegraph saved: {self._staging_dir}/{html_sub}")
        except Exception as e:  # noqa: BLE001  -- profiling must never crash the pipeline
            logger.warning(
                f"[memray] {self._id}: "
                f"failed to generate flamegraph for {self._bin_path}; "
                f".bin file is still available for offline analysis: {e}",
                exc_info=True,
            )

        self._bin_path = None

    def flush(self) -> None:
        """No-op for memray -- all artifacts are flushed per-call."""


class _GpuProfilingBackend:
    """GPU profiling backend using ``torch.profiler``.

    Captures CUDA kernel launches, operator breakdown, GPU memory
    allocations, and CPU-GPU synchronization for each
    ``process_data()`` call.  Each call produces a Chrome Trace Event
    JSON file viewable in Perfetto or ``chrome://tracing``.

    The backend is silently disabled when:

    * No CUDA device is available (checked in ``start()``, not
      ``__init__``, to avoid premature CUDA initialization before
      Ray assigns GPU resources).
    * An unrecoverable error occurs during profiling (including
      CUDA driver errors during ``stop()``).

    ``torch`` is expected to be installed in all pipeline
    environments (Docker / pixi).

    ``torch.profiler`` does **not** use ``sys.setprofile``, so there
    is no LIFO conflict with pyinstrument or memray.  It should be
    the **outermost** layer (started first, stopped last) since it
    captures CUDA events globally.

    Attributes:
        _gpu_disabled: ``True`` after an unrecoverable error or when
            CUDA is unavailable.

    """

    def __init__(
        self,
        stage_name: str,
        staging_dir: pathlib.Path,
        *,
        enabled: bool = False,
        exclude: frozenset[str] = frozenset(),
    ) -> None:
        self._stage_name = stage_name
        self._staging_dir = staging_dir
        self._call_id: str = ""
        self._profiler: Any = None
        self._gpu_disabled = not enabled or stage_name in exclude
        self._id = f"{stage_name}_{process_tag()}"

        if self._gpu_disabled and enabled:
            logger.debug(f"[torch.profiler] {self._id}: excluded by --profile-gpu-exclude")

    def start(self, call_id: str) -> None:
        """Start a ``torch.profiler.profile`` context.

        CUDA availability is checked here (not in ``__init__``) so
        that ``import torch`` and ``torch.cuda.is_available()`` run
        only when GPU profiling actually needs to begin.  Calling
        them in ``__init__`` would trigger CUDA lazy-init before Ray
        has assigned GPU resources, causing "CUDA driver error:
        initialization error" or consuming GPU memory that the
        model needs later.

        Args:
            call_id: Human-readable call identifier for file naming
                (e.g. ``"setup_on_node_1"``, ``"process_data_3"``).

        """
        if self._gpu_disabled:
            return
        self._call_id = call_id

        # Guard against leftover profiler from a failed previous attempt.
        if self._profiler is not None:
            self.stop_on_error()

        try:
            import torch  # noqa: PLC0415

            # Check CUDA availability at start time, not __init__ time.
            # torch.cuda.is_available() triggers CUDA lazy-init; doing
            # this too early (before Ray assigns GPUs) can fail or
            # consume GPU memory before the model loads.
            if not torch.cuda.is_available():
                logger.info(f"[torch.profiler] {self._id}: no CUDA device; GPU profiling disabled")
                self._gpu_disabled = True
                return

            from torch.profiler import ProfilerActivity  # noqa: PLC0415

            self._profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self._profiler.__enter__()
        except Exception as e:  # noqa: BLE001 -- profiling must never crash the pipeline
            logger.warning(
                f"[torch.profiler] {self._id}: failed to start; GPU profiling disabled for this actor: {e}",
                exc_info=True,
            )
            self.stop_on_error()
            self._gpu_disabled = True

    def stop_on_error(self) -> None:
        """Exit the profiler context without saving artifacts (error rollback)."""
        if self._profiler is None:
            return
        with contextlib.suppress(Exception):
            self._profiler.__exit__(None, None, None)
        self._profiler = None

    def stop(self) -> None:
        """Stop the profiler, dump key_averages to stdout, and export Chrome trace.

        The profiler's ``__exit__`` calls ``torch.cuda.synchronize()``
        which can raise ``RuntimeError`` if the CUDA driver is broken
        or was never properly initialized.  We catch that here so the
        error is logged but does not propagate to the pipeline.
        """
        if self._profiler is None:
            return

        try:
            self._profiler.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001 -- CUDA errors must not crash the pipeline
            logger.warning(
                f"[torch.profiler] {self._id}: profiler exit failed (CUDA driver error?); GPU profiling disabled: {e}",
            )
            self._profiler = None
            self._gpu_disabled = True
            return

        prof = self._profiler
        self._profiler = None

        # Dump concise key_averages table to stdout for immediate visibility.
        try:
            logger.info(
                f"[torch.profiler] {self._id} call #{self._call_id}\n"
                f"{prof.key_averages().table(sort_by='cuda_time_total', row_limit=15)}",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[torch.profiler] {self._id}: failed to print key_averages: {e}",
                exc_info=True,
            )

        # Export per-call Chrome Trace JSON to the local staging directory.
        # torch.profiler.export_chrome_trace() requires a local path.
        base = artifact_id(self._stage_name, self._call_id)
        trace_sub = f"gpu/{base}.json"

        try:
            path = _resolve_staging_path(self._staging_dir, trace_sub)
            prof.export_chrome_trace(str(path))
            logger.debug(f"[torch.profiler] {self._id}: saved: {self._staging_dir}/{trace_sub}")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[torch.profiler] {self._id}: failed to export trace: {e}",
                exc_info=True,
            )

    def flush(self) -> None:
        """No-op -- each call produces its own trace file.

        Present for interface symmetry with other backends.
        """


class _ProfilingState:
    """Per-actor instrumentation state for pipeline stages.

    Composes ``_CpuProfilingBackend``, ``_MemoryProfilingBackend``,
    and ``_GpuProfilingBackend`` into a single orchestrator.  Each instrumented stage actor gets exactly one
    ``_ProfilingState`` instance (created lazily on the first
    ``process_data()`` call).

    Start / stop order follows LIFO nesting so that
    ``sys.setprofile`` hooks installed by pyinstrument and memray
    are unwound correctly::

        start():          pyinstrument (outermost) -> memray -> gpu (innermost)
        stop_and_save():  gpu -> memray -> pyinstrument (outermost)

    Pyinstrument starts first so it captures the full picture,
    including any overhead from memray.  GPU profiler
    (torch.profiler) is innermost because it captures CUDA events
    globally and does NOT use ``sys.setprofile``, so it has no
    LIFO conflict.

    Adding a new backend requires creating a class with
    ``start(call_count)``, ``stop()``, ``stop_on_error()``, and
    ``flush()`` methods, then wiring it into this orchestrator.
    No stage code changes are needed.

    Attributes:
        _stage_name: Stage name for log messages.
        _cpu: CPU profiling backend (pyinstrument).
        _mem: Memory profiling backend (memray).
        _gpu: GPU profiling backend (torch.profiler).
        _call_count: Monotonically increasing counter for unique
            file names, shared across backends.

    """

    def __init__(self, stage_name: str, config: ProfilingConfig) -> None:
        self._stage_name = stage_name
        self._call_count = 0
        self._id = f"{stage_name}_{process_tag()}"

        # Resolve the base staging directory.  Three sources, in
        # priority order:
        #
        #   1. config.staging_dir -- serialized with the stage from
        #      the driver; works even when the worker doesn't inherit
        #      the driver's env vars (e.g. ray job submit in NVCF).
        #   2. COSMOS_CURATOR_ARTIFACTS_STAGING_DIR env var -- set by
        #      ArtifactDelivery.create() on the driver.  Available
        #      on workers that DO inherit env vars (standard ray.init
        #      on the same node).
        #   3. Temp directory -- fallback when neither is available
        #      (e.g. running without ArtifactDelivery).
        #
        # Once resolved, set the env var so that downstream code
        # (setup_tracing, other backends) can read it.
        base_staging_str = (
            config.staging_dir
            or os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", "")
            or tempfile.mkdtemp(prefix="cosmos_curator_profiling_staging_")
        )
        os.environ.setdefault("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", base_staging_str)
        base_staging = pathlib.Path(base_staging_str)

        # Ensure OTel tracing is initialised on this worker process.
        # The primary path is Ray's _tracing_startup_hook, but that
        # hook is a private API and may silently fail (unsupported Ray
        # version, import error in a pixi env, etc.).  Calling
        # setup_tracing() here acts as a belt-and-suspenders fallback.
        # The re-entrancy guard inside setup_tracing() makes this a
        # no-op if the Ray hook already ran.
        #
        # COSMOS_CURATOR_ARTIFACTS_STAGING_DIR is already set (above),
        # so TracingConfig.from_env() inside setup_tracing() will
        # derive trace_dir = <staging>/traces/ automatically.
        if config.tracing_enabled:
            with contextlib.suppress(Exception):
                from cosmos_curator.core.utils.infra.tracing_hook import setup_tracing  # noqa: PLC0415

                setup_tracing()

        # The "profiling" subdirectory isolates profiling artifacts
        staging_dir = base_staging / "profiling"
        staging_dir.mkdir(parents=True, exist_ok=True)

        self._cpu = _CpuProfilingBackend(
            stage_name,
            staging_dir,
            enabled=config.cpu_enabled,
            exclude=config.cpu_exclude,
        )
        self._mem = _MemoryProfilingBackend(
            stage_name,
            staging_dir,
            enabled=config.memory_enabled,
            exclude=config.memory_exclude,
        )
        self._gpu = _GpuProfilingBackend(
            stage_name,
            staging_dir,
            enabled=config.gpu_enabled,
            exclude=config.gpu_exclude,
        )

    def start(self, label: str = "process_data") -> None:
        """Start all enabled backends for the current call.

        Each backend handles its own errors internally: on failure it
        logs a warning and disables itself for the rest of the actor
        lifetime, so profiling never crashes the pipeline.

        Start order (outermost first): pyinstrument -> memray -> gpu.

        Args:
            label: Human-readable label for this call, included in
                artifact file names (e.g. ``"setup_on_node"``,
                ``"setup"``, ``"process_data"``).

        """
        self._call_count += 1
        call_id = f"{label}_{self._call_count}"
        self._cpu.start(call_id)
        self._mem.start(call_id)
        self._gpu.start(call_id)

    def stop_and_save(self) -> None:
        """Stop all active backends and persist their artifacts.

        LIFO order (innermost first): gpu -> memray -> pyinstrument,
        matching the reverse of ``start()`` so that ``sys.setprofile``
        hooks are unwound correctly.

        Each stop is guarded so one backend failure does not prevent
        the other from flushing its data.
        """
        try:
            self._gpu.stop()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[torch.profiler] {self._id}: error during stop: {e}", exc_info=True)
            TracedSpan.current().record_exception(e)
            self._gpu.stop_on_error()

        try:
            self._mem.stop()
        except Exception as e:  # noqa: BLE001  -- profiling must never crash the pipeline
            logger.warning(f"[memray] {self._id}: error during stop: {e}", exc_info=True)
            TracedSpan.current().record_exception(e)
            self._mem.stop_on_error()

        try:
            self._cpu.stop()
        except Exception as e:  # noqa: BLE001  -- profiling must never crash the pipeline
            logger.warning(f"[pyinstrument] {self._id}: error during stop: {e}", exc_info=True)
            TracedSpan.current().record_exception(e)
            self._cpu.stop_on_error()

    @contextlib.contextmanager
    def scope(self, label: str = "process_data") -> Generator[None, None, None]:
        """Context manager that brackets a block with start/stop.

        Wraps the block in an OTel span ``"{stage_name}.{label}"`` so
        that lifecycle calls (``setup_on_node``, ``setup``,
        ``process_data``) appear in the distributed trace with the
        stage name and label as attributes.

        The span also carries a ``profiling.artifact_id`` attribute
        matching the filename convention used by CPU / memory / GPU
        backends::

            MyStage_setup_on_node_1_epaniot-pc_6135

        This allows correlating a trace span with its corresponding
        profiling files (``cpu/<id>.html``, ``memory/<id>.bin``, etc.).

        Callers can enrich the span with call-site-specific attributes
        via ``TracedSpan.current().set_attribute(...)`` inside the block.

        Equivalent to calling ``start(label)`` before the block,
        ``stop_and_save()`` in a ``finally`` clause, and
        ``flush_final_artifacts()`` to persist any remaining
        backend artifacts.

        Args:
            label: Human-readable label forwarded to ``start()``.

        """
        # Build the artifact ID before incrementing the counter inside
        # start(), so the count matches what start() will use.
        call_id = f"{label}_{self._call_count + 1}"
        prof_artifact_id = artifact_id(self._stage_name, call_id)

        # Flush tracing after the traced_span exits so the span
        # itself is included in the flushed data.  The flush is
        # deferred to the outer finally block via a flag, mirroring
        # the pattern already used in destroy() (below).
        #
        # Note: flush_tracing() no longer closes the file handle --
        # it only pushes data to the OS kernel buffer.  The file
        # stays open for any late-arriving spans.  shutdown() (via
        # atexit) closes the file after disabling the provider.
        _needs_trace_flush = False
        try:
            with traced_span(
                f"{self._stage_name}.{label}",
                attributes={
                    "stage.name": self._stage_name,
                    "stage.lifecycle": label,
                    "profiling.artifact_id": prof_artifact_id,
                },
            ):
                # OTel's start_as_current_span automatically records
                # unhandled exceptions and sets ERROR status (defaults:
                # record_exception=True, set_status_on_exception=True).
                # No explicit except/record_exception needed here.
                self.start(label)
                try:
                    yield
                except BaseException:
                    # On unhandled error, signal that the OTel trace file
                    # should be flushed to the staging directory.  For
                    # setup failures, destroy() is never called and Ray
                    # kills workers with SIGKILL (atexit never runs), so
                    # the outer finally block is the only opportunity to
                    # persist worker spans to the staging dir.
                    # flush_tracing() is idempotent -- if destroy() does
                    # run later, its call will be a no-op.
                    _needs_trace_flush = True
                    raise
                finally:
                    self.stop_and_save()
                    self.flush_final_artifacts()
        finally:
            if _needs_trace_flush:
                try:
                    from cosmos_curator.core.utils.infra.tracing_hook import flush_tracing  # noqa: PLC0415

                    flush_tracing()
                except Exception as e:  # noqa: BLE001 -- profiling must never crash the pipeline
                    logger.debug(
                        f"[otel] {self._stage_name}_{process_tag()}: flush_tracing() failed on exception path: {e}",
                    )

    def flush_final_artifacts(self) -> None:
        """Flush end-of-life artifacts for all backends.

        Called once per actor during ``destroy()`` to produce final
        summary reports (e.g. combined CPU flame-tree across all
        ``process_data()`` calls).
        """
        self._cpu.flush()
        self._mem.flush()
        self._gpu.flush()


# Dynamic subclass factory and wrapper


def _make_profiled_stage_class[T: CuratorStage](
    stage_cls: type[T],
    config: ProfilingConfig,
) -> type[T]:
    """Create a dynamic subclass that wraps ``process_data()`` with instrumentation.

    The returned class is a subclass of *stage_cls* and overrides
    ``process_data()`` to call ``_ProfilingState.start()`` before and
    ``_ProfilingState.stop_and_save()`` after the real implementation.

    A ``_ProfilingState`` is lazily created on the first call so that
    it lives inside the Ray actor process (not on the driver).

    The ``destroy()`` method is also overridden to flush any remaining
    instrumentation artifacts (e.g. combined CPU report).

    Args:
        stage_cls: The base stage class to wrap.
        config: Frozen ``ProfilingConfig`` with enabled backends.

    Returns:
        A new stage class with instrumentation hooks.

    """
    base_name = stage_cls.__name__

    class _ProfiledStage(stage_cls):  # type: ignore[valid-type, misc]
        _profiling_state: _ProfilingState | None = None
        _profiling_config: ProfilingConfig = config

        def _ensure_state(self) -> _ProfilingState:
            """Lazy-init the profiling state inside the actor process.

            On first call, attaches the driver's trace anchor as the remote
            parent context so that all subsequent ``scope()`` spans become
            children of the pipeline's root trace.
            """
            if self._profiling_state is None:
                from cosmos_curator.core.utils.infra.tracing_hook import (  # noqa: PLC0415
                    attach_remote_parent,
                    read_propagated_traceparent,
                )

                traceparent = self._profiling_config.traceparent or read_propagated_traceparent()
                attach_remote_parent(traceparent)

                self._profiling_state = _ProfilingState(
                    stage_name=base_name,
                    config=self._profiling_config,
                )
            return self._profiling_state

        # stage_setup_on_node() and stage_setup() run once per actor and can
        # be expensive (model weight copies, GPU context init, etc.).
        # They are profiled via the same _ProfilingState as process_data()
        # so all backends (cpu, memory, gpu) capture the cost.
        #
        # File naming includes the caller label and a sequential counter:
        #   <Stage>_setup_on_node_1_host_pid.html
        #   <Stage>_setup_2_host_pid.html
        #   <Stage>_process_data_3_host_pid.html

        def stage_setup_on_node(self) -> None:
            """Profile the per-node setup (weight copies, etc.)."""
            with self._ensure_state().scope("setup_on_node"):
                super().stage_setup_on_node()

        def stage_setup(self) -> None:
            """Profile the per-actor setup (model loading, GPU init, etc.)."""
            with self._ensure_state().scope("setup"):
                super().stage_setup()

        def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask] | None:
            """Wrap ``process_data()`` with instrumentation hooks.

            The OTel span is created inside ``scope()``; here we only
            enrich it with the task count so per-sample child spans
            from ``StageTimer.time_process()`` nest correctly.
            """
            with self._ensure_state().scope("process_data"):
                TracedSpan.current().set_attribute("stage.num_input_tasks", len(tasks))
                return super().process_data(tasks)  # type: ignore[no-any-return]

        def destroy(self) -> None:
            """Flush remaining instrumentation artifacts, then delegate to real destroy.

            After flushing per-stage backends (cpu/memory/gpu), also
            flushes the per-process OTel trace file via
            ``flush_tracing()``.  This pushes buffered span data to the
            OS kernel so it survives SIGKILL.  The file handle stays
            **open** so late-arriving spans can still be exported; only
            ``shutdown()`` (via ``atexit``) closes it after disabling
            the provider.

            ``flush_tracing()`` is idempotent -- calling it multiple
            times is safe and cheap (just re-flushes the buffer).
            """
            with traced_span(
                f"{base_name}.destroy",
                attributes={"stage.name": base_name, "stage.lifecycle": "destroy"},
            ):
                if self._profiling_state is not None:
                    self._profiling_state.flush_final_artifacts()
                super().destroy()

            # Flush the per-process OTel trace file to the staging
            # directory AFTER the traced_span block has closed (so the
            # destroy span itself is included in the persisted file).
            # Imported here to avoid pulling tracing_hook's deps
            # at module level in environments where they may be
            # unavailable.
            try:
                from cosmos_curator.core.utils.infra.tracing_hook import flush_tracing  # noqa: PLC0415

                flush_tracing()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[otel] {base_name}_{process_tag()}: flush_tracing() failed: {e}")

    _ProfiledStage.__name__ = base_name
    _ProfiledStage.__qualname__ = base_name
    return _ProfiledStage


def profiling_wrapper(
    stage: CuratorStage,
    config: ProfilingConfig,
) -> CuratorStage:
    """Wrap a stage's ``process_data()`` with instrumentation.

    Swaps the instance's class in place (same approach as
    ``stage_save_wrapper`` in ``stage_replay.py``) so that all existing
    attributes are preserved.  The new class is a dynamic subclass
    that delegates to ``_ProfilingState`` around every
    ``process_data()`` call.

    Args:
        stage: The stage instance to instrument.
        config: Frozen ``ProfilingConfig`` with enabled backends.

    Returns:
        The same *stage* instance with its ``__class__`` swapped.

    """
    logger.debug(
        f"[profiling] {stage.__class__.__name__}_{process_tag()}: wrapping "
        f"(cpu={'ON' if config.cpu_enabled else 'OFF'}, "
        f"mem={'ON' if config.memory_enabled else 'OFF'}, "
        f"gpu={'ON' if config.gpu_enabled else 'OFF'}, "
        f"dir={config.profile_dir})",
    )
    stage.__class__ = _make_profiled_stage_class(stage.__class__, config)
    return stage


def _parse_exclude(raw: str | None) -> frozenset[str]:
    """Parse a comma-separated exclude string into a frozenset of scope names.

    Args:
        raw: Comma-separated string (e.g. ``"_root,MyStage"``)
            or ``None``.

    Returns:
        A frozenset of stripped, non-empty scope names.

    """
    if not raw:
        return frozenset()
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def _apply_profiling_config(args: argparse.Namespace) -> ProfilingConfig | None:
    """Build a ``ProfilingConfig`` from parsed CLI arguments and apply side effects.

    Returns ``None`` if no instrumentation flags were requested,
    meaning zero overhead is added to the pipeline.

    Side effect: any instrumentation flag (``--profile-cpu``,
    ``--profile-memory``, ``--profile-gpu``, ``--profile-tracing``)
    implies ``--perf-profile``, so ``args.perf_profile`` is set to
    ``True`` when any profiling backend is enabled.  This ensures
    ``StageTimer.log_stats()`` is called -- the framework gate
    for flushing per-task metrics.

    The profiling base directory is the static subdirectory
    ``profile/`` under the pipeline output path, for simplicity --
    no configurable CLI override exists.  The output path is
    auto-discovered from *args* by probing common attributes
    (``output_clip_path``, ``output_dataset_path``,
    ``output_prefix``).  Falls back to ``./profile`` when none is
    found.  Backend subdirectories:

    * ``<output-path>/profile/cpu/``      -- pyinstrument
    * ``<output-path>/profile/memory/``   -- memray
    * ``<output-path>/profile/gpu/``      -- torch.profiler
    * ``<output-path>/profile/traces/``   -- OpenTelemetry (Ray distributed tracing)

    The resolved base directory is stored in
    ``ProfilingConfig.profile_dir`` so that backends can derive
    their own subdirectories from it.

    Args:
        args: Parsed CLI namespace (must contain ``profile_cpu``,
            ``profile_memory``, ``profile_gpu``,
            and ``perf_profile`` attributes).

    Returns:
        A frozen ``ProfilingConfig`` or ``None``.

    """
    cpu = getattr(args, "profile_cpu", False)
    mem = getattr(args, "profile_memory", False)
    gpu = getattr(args, "profile_gpu", False)
    tracing = getattr(args, "profile_tracing", False)

    any_profiling = cpu or mem or gpu or tracing
    if not any_profiling and not getattr(args, "perf_profile", False):
        return None

    # Discover the pipeline output directory from common args attributes
    # and derive the profiling base path as <output_path>/profile.
    output_dir: str | None = None
    for attr in ("output_clip_path", "output_dataset_path", "output_prefix"):
        candidate = getattr(args, attr, None)
        if candidate is not None:
            output_dir = candidate
            break

    # Use string concatenation instead of pathlib.Path to preserve
    # URI schemes (e.g. "s3://bucket/key/profile" -- pathlib normalizes
    # the double-slash to a single slash, corrupting the URL).
    profile_dir = f"{output_dir.rstrip('/')}/profile" if output_dir is not None else "./profile"

    # Any profiling flag implies --perf-profile (so log_stats() is invoked).
    if any_profiling and not args.perf_profile:
        logger.info(
            f"[profiling] {process_tag()}: "
            f"--profile-cpu/--profile-memory/--profile-gpu implies --perf-profile; enabling",
        )
        args.perf_profile = True

    return ProfilingConfig(
        profile_dir=profile_dir,
        s3_profile_name=getattr(args, "output_s3_profile_name", "default"),
        cpu_enabled=cpu,
        memory_enabled=mem,
        gpu_enabled=gpu,
        tracing_enabled=tracing,
        tracing_sampling=getattr(args, "profile_tracing_sampling", 0.01),
        tracing_otlp_endpoint=getattr(args, "profile_tracing_otlp_endpoint", ""),
        staging_dir=os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", ""),
        traceparent=os.environ.get("COSMOS_CURATOR_TRACEPARENT", ""),
        cpu_exclude=_parse_exclude(getattr(args, "profile_cpu_exclude", None)),
        memory_exclude=_parse_exclude(getattr(args, "profile_memory_exclude", None)),
        gpu_exclude=_parse_exclude(getattr(args, "profile_gpu_exclude", None)),
    )


def _register_flush_hooks(config: ProfilingConfig, *, has_profiling: bool, state: "_ProfilingState") -> None:
    """Register pre-shutdown hooks that flush artifacts before collection.

    Called from :func:`profiling_scope` after ``ArtifactDelivery``
    hooks have been registered.  LIFO ordering ensures these flush
    hooks run *before* the collection hooks.

    Args:
        config: Current profiling configuration.
        has_profiling: Whether CPU/memory/GPU profiling is enabled.
        state: The ``_ProfilingState`` whose root profilers must be
            flushed before artifact collection.

    """
    from cosmos_curator.core.utils.infra.ray_cluster_utils import register_pre_shutdown_hook  # noqa: PLC0415

    if has_profiling:

        def _flush_root_profilers() -> None:
            state.stop_and_save()
            state.flush_final_artifacts()

        register_pre_shutdown_hook(_flush_root_profilers)

    if config.tracing_enabled:

        def _flush_driver_traces() -> None:
            try:
                from cosmos_curator.core.utils.infra.tracing_hook import flush_tracing  # noqa: PLC0415

                flush_tracing()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[otel] {process_tag()}: Failed to flush driver traces: {e}")

        register_pre_shutdown_hook(_flush_driver_traces)


@contextlib.contextmanager
def profiling_scope(
    args: argparse.Namespace,
    *,
    stage_name: str = "_root",
    label: str = "main",
) -> Generator[None, None, None]:
    r"""Context manager that profiles the wrapped block.

    Builds a ``ProfilingConfig`` from *args* via
    ``_apply_profiling_config()`` and, if any profiling is enabled,
    activates the configured backends for the lifetime of the
    ``with`` block.

    When profiling backends are enabled (cpu/memory/gpu),
    ``ArtifactDelivery`` is created (``kind="profiling"``) to
    collect artifacts from ``<staging>/profiling/`` post-pipeline.

    When ``--profile-tracing`` is enabled, Ray is pre-initialized
    with the OpenTelemetry tracing hook before yielding.  Each
    worker flushes spans to a local staging directory.  A second
    ``ArtifactDelivery`` (``kind="traces"``) collects trace files
    from ``<staging>/traces/`` post-pipeline.  By default only the
    local file exporter is active; pass
    ``--profile-tracing-otlp-endpoint`` to also send spans to a
    remote OTLP collector.

    Both delivery instances register pre-shutdown hooks independently
    and upload to ``<output-path>/profile`` via ``StorageWriter``.

    Which per-stage backends actually run is controlled by the global
    ``--profile-cpu`` / ``--profile-memory`` / ``--profile-gpu``
    flags **and** the per-backend ``--profile-*-exclude`` flags.
    For example, to profile the driver with CPU only::

        --profile-cpu --profile-memory \
          --profile-memory-exclude=_root

    Typical use is at the CLI entry point::

        with profiling_scope(args):
            args.func(args)

    No-op when no profiling flags are present, so callers can use
    it unconditionally.

    Args:
        args: Parsed CLI namespace (forwarded to
            ``_apply_profiling_config()``).
        stage_name: Logical name for the profiling scope, used in
            artifact file names and exclude matching
            (default ``"_root"``).
        label: Human-readable label for the call, forwarded to
            ``_ProfilingState.scope()`` (default ``"main"``).

    Yields:
        Nothing.  Profilers run for the lifetime of the block.

    """
    config = _apply_profiling_config(args)
    if config is None:
        yield
        return

    # Set up artifact delivery BEFORE the pipeline so that workers
    # inherit the COSMOS_CURATOR_ARTIFACTS_STAGING_DIR environment variable.
    # ArtifactDelivery.create() registers collect() as a Ray
    # pre-shutdown hook so artifact collection runs automatically
    # before ray.shutdown() while the cluster is still alive.
    #
    # Each subsystem gets its own ArtifactDelivery instance with a
    # distinct ``kind`` (staging subdirectory) and ``upload_subdir``:
    #   - "profiling": cpu/memory/gpu artifacts -> <output-path>/profile/
    #   - "traces":    OTel span files         -> <output-path>/profile/traces/
    has_profiling = config.cpu_enabled or config.memory_enabled or config.gpu_enabled
    if has_profiling:
        ArtifactDelivery.create(
            kind="profiling",
            output_dir=config.profile_dir,
            s3_profile_name=config.s3_profile_name,
        )

    # Set up distributed tracing (OTel via Ray) if requested.
    # This must happen BEFORE the pipeline starts so that ray.init()
    # is called with the tracing hook before xenna's ray.init().
    # Each worker flushes spans to the staging directory via
    # flush_tracing() (called from _ProfiledStage.destroy()) with
    # atexit as fallback.
    #
    # IMPORTANT: ArtifactDelivery for traces MUST be created BEFORE
    # enable_tracing().  enable_tracing() reads
    # COSMOS_CURATOR_ARTIFACTS_STAGING_DIR to determine the trace
    # directory and propagates it to workers via COSMOS_CURATOR_TRACE_DIR.
    # If the staging dir env var is not yet set (e.g. when only tracing
    # is enabled, no cpu/mem/gpu), enable_tracing() falls back to
    # /tmp/cosmos_curator_traces/ -- which doesn't match the staging
    # directory that ArtifactDelivery would create later, causing
    # collection to find zero files.
    # Globally disable or enable the OTel SDK.  This must happen before
    # ray.init() so workers inherit the env var.  When tracing is disabled,
    # OTEL_SDK_DISABLED=true tells all OTel-aware libraries (vLLM, boto
    # instrumentors, etc.) to produce no-op spans.  When tracing is
    # enabled, we clear it so the SDK is active.
    if config.tracing_enabled:
        os.environ.pop("OTEL_SDK_DISABLED", None)
        try:
            # Create the traces ArtifactDelivery first so that
            # COSMOS_CURATOR_ARTIFACTS_STAGING_DIR is set before
            # enable_tracing() reads it.
            if config.profile_dir:
                ArtifactDelivery.create(
                    kind="traces",
                    output_dir=config.profile_dir,
                    s3_profile_name=config.s3_profile_name,
                    upload_subdir="traces",
                )

            from cosmos_curator.core.utils.infra.tracing_hook import enable_tracing  # noqa: PLC0415

            enable_tracing(
                sampling_rate=config.tracing_sampling,
                otlp_endpoint=config.tracing_otlp_endpoint,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[otel] {process_tag()}: Failed to set up distributed tracing: {e}", exc_info=True)
    else:
        os.environ["OTEL_SDK_DISABLED"] = "true"

    # Run the per-stage profiling scope (cpu/memory/gpu backends).
    # scope() wraps the block with an OTel span automatically.
    state = _ProfilingState(stage_name=stage_name, config=config)

    # Register pre-shutdown hooks that flush artifacts to the staging
    # directory BEFORE ArtifactDelivery collection hooks run.
    # Without these, artifacts may be written too late:
    #
    #   The pipeline calls ``shutdown_cluster`` inside the yield.
    #   ``scope.__exit__`` (which writes root files) runs AFTER
    #   ``shutdown_cluster`` returns -- by which point ArtifactDelivery
    #   has already collected and Ray has been shut down.
    #
    # By registering AFTER the ArtifactDelivery hooks, LIFO ordering
    # ensures flush hooks run FIRST:
    #
    #   Shutdown sequence (LIFO):
    #     1. flush driver traces   -- flushes OTel span file to disk
    #     2. flush root profilers  -- writes cpu/mem/gpu files to staging
    #     3. collect traces        -- ArtifactDelivery uploads spans
    #     4. collect profiling     -- ArtifactDelivery uploads profiles
    #     5. Ray shutdown
    #
    # ``stop_and_save`` is idempotent (each backend guards on None
    # state), so the subsequent ``scope.__exit__`` call is a safe
    # no-op.  ``flush_tracing`` is also idempotent (flushes buffers
    # but leaves the file open for any late spans).
    _register_flush_hooks(config, has_profiling=has_profiling, state=state)

    # Trace hierarchy: the "trace anchor" is the TRUE root of the
    # distributed trace.  It has no parent, ends immediately, and is
    # exported to all backends (file + OTLP) before any child span
    # arrives.  This eliminates Jaeger's "invalid parent span ID"
    # warnings that occur when a long-lived root span arrives last.
    #
    # _root.main (created by state.scope) becomes a CHILD of the
    # anchor.  Stage spans also reference the anchor as parent (via
    # COSMOS_CURATOR_TRACEPARENT env var).
    #
    #   Trace hierarchy
    #   ---------------
    #   trace_anchor  (root, no parent, ends FIRST)
    #     +-- _root.main              (ends LAST, parent=anchor)
    #     |     +-- botocore S3 etc.  (auto-instrumented, parent=main)
    #     +-- stage spans             (parent=anchor via env var)
    #
    #   Timeline
    #   --------
    #   trace_root_anchor starts + ends -- exported to backends NOW
    #     scope("main") starts          -- _root.main span starts
    #       propagate_trace_context()   -- propagates anchor's IDs
    #       yield
    #         pipeline runs
    #         shutdown_cluster
    #           collect .jsonl          -- anchor is present
    #           Ray shuts down
    #     scope("main") ends            -- _root.main written
    #
    # Root profiler artifacts are flushed by the pre-shutdown hook
    # registered above (LIFO, runs before ArtifactDelivery.collect()).
    # scope().__exit__() calls stop_and_save() and
    # flush_final_artifacts() again, but both are idempotent no-ops
    # at this point.  No additional action needed here.
    with contextlib.ExitStack() as _anchor_stack:
        if config.tracing_enabled:
            try:
                from cosmos_curator.core.utils.infra.tracing_hook import propagate_trace_context  # noqa: PLC0415

                _anchor_stack.enter_context(
                    trace_root_anchor(f"{stage_name}.trace_anchor"),
                )
                propagate_trace_context()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[otel] {process_tag()}: Failed to set up trace anchor: {e}")

        with state.scope(label):
            # Enrich the root span with profiling configuration flags.
            TracedSpan.current().set_attributes(
                {
                    "profiling.cpu_enabled": config.cpu_enabled,
                    "profiling.memory_enabled": config.memory_enabled,
                    "profiling.gpu_enabled": config.gpu_enabled,
                    "profiling.tracing_enabled": config.tracing_enabled,
                    "profiling.tracing_sampling": config.tracing_sampling,
                }
            )

            yield
