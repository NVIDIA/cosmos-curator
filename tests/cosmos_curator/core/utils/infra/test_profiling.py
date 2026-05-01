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

"""Tests for the profiling module (``profiling.py``).

Exercises each profiling backend with **real** profiling tools to
verify that the module achieves its purpose: producing artifact files
that operators can use for post-hoc performance analysis.

::

    What we test                          How we test it
    +----------------------------------+  +--------------------------------------+
    | CPU backend (pyinstrument)       |  | Real Profiler: start, sleep, stop    |
    |   .html + .pyisession on disk    |  |   -> assert files exist + valid HTML |
    | Memory backend (memray)          |  | Real Tracker: start, alloc, stop     |
    |   .bin capture on disk           |  |   -> assert .bin non-empty           |
    | GPU backend (torch.profiler)     |  | SKIPPED (requires CUDA hardware)     |
    | Excluded stages                  |  | Enable + exclude -> no artifacts     |
    | Error resilience                 |  | Monkeypatch broken ctor -> disabled  |
    | _ProfilingState orchestrator     |  | scope() -> cpu/ + memory/ artifacts  |
    | ProfilingConfig                  |  | Defaults correct, frozen immutable   |
    | _apply_profiling_config (CLI)    |  | argparse.Namespace -> config or None |
    | profiling_wrapper (end-to-end)   |  | Wrap FakeStage, call process_data    |
    |                                  |  |   -> cpu/ artifacts on disk          |
    | _resolve_staging_path            |  | mkdir + stale-file removal           |
    +----------------------------------+  +--------------------------------------+

Test setup:
    ``_ProfilingState.scope()`` internally calls ``traced_span()`` from
    the tracing module, which requires a valid global OTel
    ``TracerProvider``.  The ``_ensure_valid_otel_provider`` autouse
    fixture resets OTel's set-once guard and installs a fresh no-op SDK
    provider before each test so that ``scope()`` never hits the
    recursion bug in OTel's default proxy provider.

    Staging directories use pytest's ``tmp_path`` fixture.  The env var
    ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR`` is set via ``monkeypatch``
    where needed so ``_ProfilingState`` picks up the test directory.

    No Ray cluster, no network, no GPU required.
"""

import argparse
import pathlib
import time
from collections.abc import Generator

import attrs
import pytest
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, PipelineTask
from cosmos_curator.core.utils.infra.profiling import (
    ProfilingConfig,
    _apply_profiling_config,
    _CpuProfilingBackend,
    _MemoryProfilingBackend,
    _ProfilingState,
    _resolve_staging_path,
    profiling_wrapper,
)


@pytest.fixture(autouse=True)
def _ensure_valid_otel_provider() -> None:
    """Install a fresh no-op OTel TracerProvider before every test.

    ``_ProfilingState.scope()`` calls ``traced_span()`` which calls
    ``trace.get_tracer()``.  Without a valid SDK provider the default
    proxy provider may recurse infinitely.  Resetting the set-once
    guard and installing a fresh ``TracerProvider()`` avoids this.
    """
    if hasattr(_otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        _otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
    _otel_trace.set_tracer_provider(TracerProvider())


@pytest.fixture(autouse=True)
def _hide_greenlet() -> Generator[None, None, None]:
    """Temporarily hide greenlet from ``sys.modules`` during each test.

    When the full test suite runs, earlier test files (e.g.
    ``test_database_utils``) import SQLAlchemy which pulls in greenlet.
    If memray sees greenlet in ``sys.modules`` it activates experimental
    greenlet tracking that can hang or be extremely slow on some systems.
    The profiling tests don't need greenlet, so we hide it.
    """
    import sys  # noqa: PLC0415

    saved = sys.modules.pop("greenlet", None)
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["greenlet"] = saved


class TestCpuProfilingBackend:
    """Verify that the CPU backend produces real profiling artifacts."""

    def test_start_stop_produces_html_and_session(self, tmp_path: pathlib.Path) -> None:
        """A start/stop cycle with real code produces .html and .pyisession files."""
        backend = _CpuProfilingBackend(
            "TestStage",
            tmp_path,
            enabled=True,
        )
        backend.start("process_data_1")
        # Execute a small workload so pyinstrument captures something.
        time.sleep(0.01)
        backend.stop()

        html_files = list(tmp_path.rglob("cpu/*.html"))
        session_files = list(tmp_path.rglob("cpu/*.pyisession"))
        assert len(html_files) == 1, f"Expected 1 HTML file, got {html_files}"
        assert len(session_files) == 1, f"Expected 1 session file, got {session_files}"

        # The HTML file should contain actual HTML content.
        html_content = html_files[0].read_text(encoding="utf-8")
        assert "<html" in html_content.lower()

    def test_excluded_stage_produces_no_artifacts(self, tmp_path: pathlib.Path) -> None:
        """When the stage is in the exclude set, no artifacts are produced."""
        backend = _CpuProfilingBackend(
            "MyStage",
            tmp_path,
            enabled=True,
            exclude=frozenset({"MyStage"}),
        )
        backend.start("process_data_1")
        backend.stop()

        cpu_dir = tmp_path / "cpu"
        assert not cpu_dir.exists(), "Excluded stage should not create cpu/ directory"

    def test_start_on_error_disables_backend(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If pyinstrument.Profiler raises, the backend disables itself without crashing."""

        def _broken_profiler(*_a: object, **_kw: object) -> None:
            msg = "broken profiler"
            raise RuntimeError(msg)

        monkeypatch.setattr("pyinstrument.Profiler", _broken_profiler)
        backend = _CpuProfilingBackend("TestStage", tmp_path, enabled=True)
        backend.start("process_data_1")

        assert backend._cpu_disabled is True


class TestMemoryProfilingBackend:
    """Verify that the memory backend produces real .bin captures."""

    def test_start_stop_produces_bin_file(self, tmp_path: pathlib.Path) -> None:
        """A start/stop cycle with real allocations produces a .bin file."""
        backend = _MemoryProfilingBackend(
            "TestStage",
            tmp_path,
            enabled=True,
        )
        backend.start("process_data_1")
        # Perform a real allocation so memray captures something.
        _data = [i for i in range(10000)]  # noqa: C416
        backend.stop()

        bin_files = list(tmp_path.rglob("memory/*.bin"))
        assert len(bin_files) == 1, f"Expected 1 .bin file, got {bin_files}"
        assert bin_files[0].stat().st_size > 0, ".bin file should be non-empty"

    def test_excluded_stage_produces_no_artifacts(self, tmp_path: pathlib.Path) -> None:
        """When the stage is in the exclude set, no artifacts are produced."""
        backend = _MemoryProfilingBackend(
            "MyStage",
            tmp_path,
            enabled=True,
            exclude=frozenset({"MyStage"}),
        )
        backend.start("process_data_1")
        backend.stop()

        memory_dir = tmp_path / "memory"
        assert not memory_dir.exists(), "Excluded stage should not create memory/ directory"

    def test_start_on_error_disables_backend(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If memray.Tracker raises, the backend disables itself without crashing."""
        import memray  # noqa: PLC0415

        original_tracker = memray.Tracker

        def _broken_tracker(*_a: object, **_kw: object) -> None:
            msg = "broken tracker"
            raise RuntimeError(msg)

        monkeypatch.setattr("memray.Tracker", _broken_tracker)
        backend = _MemoryProfilingBackend("TestStage", tmp_path, enabled=True)
        backend.start("process_data_1")

        assert backend._mem_disabled is True
        monkeypatch.setattr("memray.Tracker", original_tracker)


class TestProfilingStateOrchestration:
    """Verify that _ProfilingState orchestrates backends and produces artifacts."""

    def test_scope_profiles_block_and_produces_artifacts(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """scope() with cpu+memory enabled produces artifacts in both subdirs."""
        # Point the staging env var to our tmp_path so _ProfilingState
        # uses it instead of creating a random temp dir.
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        config = ProfilingConfig(
            cpu_enabled=True,
            memory_enabled=True,
        )
        state = _ProfilingState("TestStage", config)

        with state.scope("process_data"):
            time.sleep(0.01)
            _data = list(range(1000))

        # Both backends should have produced artifacts under
        # <tmp_path>/profiling/cpu/ and <tmp_path>/profiling/memory/.
        profiling_dir = tmp_path / "profiling"
        cpu_files = list(profiling_dir.rglob("cpu/*.html"))
        memory_files = list(profiling_dir.rglob("memory/*.bin"))
        assert len(cpu_files) >= 1, f"Expected CPU artifacts, found {cpu_files}"
        assert len(memory_files) >= 1, f"Expected memory artifacts, found {memory_files}"

    def test_scope_increments_call_count(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successive scope() calls produce artifacts with incrementing counters."""
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        config = ProfilingConfig(cpu_enabled=True)
        state = _ProfilingState("TestStage", config)

        with state.scope("process_data"):
            time.sleep(0.01)

        with state.scope("process_data"):
            time.sleep(0.01)

        cpu_files = sorted(f.name for f in (tmp_path / "profiling").rglob("cpu/*.html"))
        # Filenames should contain _1 and _2 for the two calls.
        assert len(cpu_files) == 2
        assert any("_1" in name for name in cpu_files)
        assert any("_2" in name for name in cpu_files)


class TestProfilingConfig:
    """Verify ProfilingConfig defaults and immutability."""

    def test_defaults(self) -> None:
        """All booleans default to False, excludes are empty, dir is './profile'."""
        config = ProfilingConfig()
        assert config.cpu_enabled is False
        assert config.memory_enabled is False
        assert config.gpu_enabled is False
        assert config.tracing_enabled is False
        assert config.cpu_exclude == frozenset()
        assert config.memory_exclude == frozenset()
        assert config.gpu_exclude == frozenset()
        assert config.profile_dir == "./profile"

    def test_frozen(self) -> None:
        """Assigning to an attribute on a frozen config raises an error."""
        config = ProfilingConfig()
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.cpu_enabled = True  # type: ignore[misc]


class TestBuildProfilingConfig:
    """Verify _apply_profiling_config parses CLI args correctly."""

    def test_no_flags_returns_none(self) -> None:
        """With no profiling flags, the function returns None (zero overhead)."""
        args = argparse.Namespace(
            profile_cpu=False,
            profile_memory=False,
            profile_gpu=False,
            profile_tracing=False,
            perf_profile=False,
        )
        assert _apply_profiling_config(args) is None

    def test_cpu_flag_returns_config_and_enables_perf(self) -> None:
        """--profile-cpu enables cpu_enabled and forces perf_profile=True."""
        args = argparse.Namespace(
            profile_cpu=True,
            profile_memory=False,
            profile_gpu=False,
            profile_tracing=False,
            perf_profile=False,
            output_clip_path="/output/clips",
        )
        config = _apply_profiling_config(args)

        assert config is not None
        assert config.cpu_enabled is True
        assert args.perf_profile is True
        assert config.profile_dir == "/output/clips/profile"

    def test_s3_output_path_preserves_scheme(self) -> None:
        """An S3 output path must keep the s3:// scheme intact in profile_dir.

        pathlib.Path normalizes "s3://bucket" to "s3:/bucket" -- the
        config builder must avoid this by using string concatenation.
        """
        args = argparse.Namespace(
            profile_cpu=True,
            profile_memory=False,
            profile_gpu=False,
            profile_tracing=False,
            perf_profile=False,
            output_clip_path="s3://my-bucket/output/run-42",
        )
        config = _apply_profiling_config(args)

        assert config is not None
        assert config.profile_dir == "s3://my-bucket/output/run-42/profile"

    def test_trailing_slash_on_output_path_is_normalized(self) -> None:
        """A trailing slash on the output path must not produce a double-slash."""
        args = argparse.Namespace(
            profile_cpu=True,
            profile_memory=False,
            profile_gpu=False,
            profile_tracing=False,
            perf_profile=False,
            output_clip_path="/output/clips/",
        )
        config = _apply_profiling_config(args)

        assert config is not None
        assert config.profile_dir == "/output/clips/profile"


class TestProfilingWrapper:
    """Verify profiling_wrapper swaps the class and preserves identity."""

    def test_wrapped_stage_name_preserved(self) -> None:
        """After wrapping, __class__.__name__ is still the original stage name."""

        class FakeStage(CuratorStage):
            pass

        stage = FakeStage()
        config = ProfilingConfig(cpu_enabled=True)
        profiling_wrapper(stage, config)

        assert stage.__class__.__name__ == "FakeStage"

    def test_wrapped_stage_process_data_calls_profiling(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calling process_data on a wrapped stage produces CPU artifacts."""
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        class FakeStage(CuratorStage):
            def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask] | None:
                time.sleep(0.01)
                return tasks

        stage = FakeStage()
        config = ProfilingConfig(cpu_enabled=True)
        profiling_wrapper(stage, config)

        # Call process_data to trigger profiling.
        stage.process_data([])

        cpu_files = list(tmp_path.rglob("profiling/cpu/*.html"))
        assert len(cpu_files) >= 1, f"Expected CPU artifacts after process_data, found {cpu_files}"


class TestScopeExceptionPath:
    """Verify scope() handles exceptions correctly with deferred trace flush.

    When an exception occurs inside scope(), the OTel trace file is
    flushed (buffered data pushed to OS kernel) after the traced_span
    exits.  The file handle stays **open** so late-arriving spans can
    still be exported; only ``shutdown()`` (via atexit) closes it.

    ::

        scope("setup")
        |
        +-- try:
        |     with traced_span("Stage.setup"):
        |       yield  --> exception raised here
        |       except BaseException:
        |         _needs_trace_flush = True   <-- flag set
        |         raise
        |       finally:
        |         stop_and_save()
        |
        +-- traced_span exits --> span exported to OPEN file (OK)
        |
        +-- finally:
              if _needs_trace_flush:
                flush_tracing()  <-- flushes buffer, file stays OPEN
    """

    @staticmethod
    def _raise_inside_scope(state: _ProfilingState, label: str, msg: str) -> None:
        """Enter scope() and immediately raise RuntimeError."""
        with state.scope(label):
            raise RuntimeError(msg)

    def test_scope_exception_calls_flush_tracing(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On exception, scope() calls flush_tracing() to persist spans."""
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        import cosmos_curator.core.utils.infra.tracing_hook as hook_mod  # noqa: PLC0415

        flush_called = False
        original_flush = hook_mod.flush_tracing

        def mock_flush() -> None:
            nonlocal flush_called
            flush_called = True
            original_flush()

        monkeypatch.setattr(hook_mod, "flush_tracing", mock_flush)

        config = ProfilingConfig(cpu_enabled=True)
        state = _ProfilingState("TestStage", config)

        with pytest.raises(RuntimeError, match="stage setup failed"):
            self._raise_inside_scope(state, "setup", "stage setup failed")

        assert flush_called, "flush_tracing() should be called on exception path"

    def test_scope_normal_path_does_not_call_flush_tracing(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On normal exit, scope() does NOT call flush_tracing().

        flush_tracing() is only for the error path (setup failures
        where destroy() never runs).  On normal exit, destroy() handles
        the flush.
        """
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        flush_called = False

        def mock_flush() -> None:
            nonlocal flush_called
            flush_called = True

        import cosmos_curator.core.utils.infra.tracing_hook as hook_mod  # noqa: PLC0415

        monkeypatch.setattr(hook_mod, "flush_tracing", mock_flush)

        config = ProfilingConfig(cpu_enabled=True)
        state = _ProfilingState("TestStage", config)

        with state.scope("process_data"):
            time.sleep(0.01)

        assert not flush_called, "flush_tracing() should NOT be called on normal path"

    def test_scope_exception_preserves_original_exception(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The original exception propagates unchanged through scope().

        The try/finally structure must not swallow, wrap, or alter the
        exception raised inside the scope block.
        """
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        config = ProfilingConfig(cpu_enabled=True)
        state = _ProfilingState("TestStage", config)

        with pytest.raises(RuntimeError, match="vllm serve exited with code 1"):
            self._raise_inside_scope(state, "setup", "vllm serve exited with code 1")

    def test_scope_exception_with_flush_tracing_failure_still_propagates(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If flush_tracing() itself fails, the original exception still propagates.

        The outer finally block catches flush failures so they don't
        mask the real error (e.g. stage setup failure).
        """
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        def broken_flush() -> None:
            msg = "trace flush exploded"
            raise OSError(msg)

        import cosmos_curator.core.utils.infra.tracing_hook as hook_mod  # noqa: PLC0415

        monkeypatch.setattr(hook_mod, "flush_tracing", broken_flush)

        config = ProfilingConfig(cpu_enabled=True)
        state = _ProfilingState("TestStage", config)

        with pytest.raises(RuntimeError, match="real setup error"):
            self._raise_inside_scope(state, "setup", "real setup error")

    def test_scope_exception_produces_span_data_before_flush(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On exception, the traced_span is exported and flush pushes data to disk.

        Sets up a real TracerProvider with a ConsoleSpanExporter so we
        can verify that the span data appears in the .jsonl file.
        flush_tracing() flushes the buffer without closing the file;
        the file is then closed manually after the test to read it.
        """
        from opentelemetry.sdk.trace import TracerProvider as SdkProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor  # noqa: PLC0415

        import cosmos_curator.core.utils.infra.tracing_hook as hook_mod  # noqa: PLC0415

        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))

        trace_dir = tmp_path / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        span_file = trace_dir / "test_spans.jsonl"
        file_handle = span_file.open("w")

        import os  # noqa: PLC0415

        if hasattr(_otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
            _otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
        provider = SdkProvider()
        provider.add_span_processor(
            SimpleSpanProcessor(
                ConsoleSpanExporter(
                    out=file_handle,
                    formatter=lambda span: span.to_json(indent=None) + os.linesep,
                ),
            ),
        )
        _otel_trace.set_tracer_provider(provider)

        # Mock flush_tracing to flush (but NOT close) the file,
        # matching the real lifecycle where flush() keeps the file open.
        def flush_without_close() -> None:
            provider.force_flush()
            file_handle.flush()

        monkeypatch.setattr(hook_mod, "flush_tracing", flush_without_close)

        config = ProfilingConfig(cpu_enabled=True)
        state = _ProfilingState("TestStage", config)

        with pytest.raises(RuntimeError, match="setup boom"):
            self._raise_inside_scope(state, "setup", "setup boom")

        # Close the file so we can read its contents.
        file_handle.close()

        content = span_file.read_text(encoding="utf-8")
        assert "TestStage.setup" in content, (
            f"Span 'TestStage.setup' should appear in the trace file. "
            f"This proves the span was exported to the still-open file "
            f"and flush pushed the data to disk. "
            f"File content: {content[:500]}"
        )


class TestResolveStagingPath:
    """Verify _resolve_staging_path creates dirs and removes stale files."""

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        """Resolving a nested path creates the parent directories."""
        result = _resolve_staging_path(tmp_path, "cpu/deeply/nested/report.html")
        assert result.parent.exists()
        assert result == tmp_path / "cpu" / "deeply" / "nested" / "report.html"

    def test_removes_stale_file(self, tmp_path: pathlib.Path) -> None:
        """If a file already exists at the path, it is removed."""
        stale = tmp_path / "cpu" / "report.html"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("old data", encoding="utf-8")

        result = _resolve_staging_path(tmp_path, "cpu/report.html")
        assert result == stale
        assert not stale.exists(), "Stale file should have been removed"
