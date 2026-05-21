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

"""Tests for the OTel tracing public API (``tracing.py``).

Verifies the core contract of the tracing module: the Null Object
pattern for ``TracedSpan``, real span creation via the OTel SDK, and
deterministic artifact naming.

::

    What we test                   How we test it
    +--------------------------+   +-------------------------------------+
    | TracedSpan(None) no-ops  |   | Call every method, assert no raise  |
    | TracedSpan(real) delgtn  |   | MagicMock span, assert forwarding   |
    | traced_span / @traced    |   | Real TracerProvider + in-mem export |
    | trace_root_anchor        |   | Real TracerProvider, verify type    |
    | artifact_id / process_tag|   | Assert format includes PID/host    |
    +--------------------------+   +-------------------------------------+

Test setup:
    OTel's global TracerProvider has a set-once guard that prevents
    overriding after the first ``set_tracer_provider()`` call.  The
    ``_ensure_valid_otel_provider`` autouse fixture resets this guard
    before and after every test so each test starts with a clean
    provider.  Tests that need to capture exported spans use the
    ``otel_provider`` fixture which installs a real ``TracerProvider``
    backed by ``_InMemoryExporter`` (a lightweight in-process span
    collector).

    No Ray cluster, no network, no GPU required.
"""

import asyncio
import contextvars
import os
from collections.abc import Generator, Sequence
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from cosmos_curator.core.utils.infra.tracing import (
    StatusCode,
    TracedSpan,
    artifact_id,
    process_tag,
    trace_root_anchor,
    traced,
    traced_span,
)


class _InMemoryExporter(SpanExporter):
    """Minimal in-memory span exporter for testing.

    Collects finished spans in a list so tests can assert on span
    names, attributes, and count without needing a real backend.
    """

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._spans)

    def shutdown(self) -> None:
        pass


def _reset_otel_provider_guard() -> None:
    """Reset OTel's set-once guard so tests can swap providers freely.

    OTel's global TracerProvider has a set-once guard that prevents
    overriding.  In tests we need to set different providers for
    different test cases.  This helper resets the internal flag.
    """
    from opentelemetry import trace as _trace  # noqa: PLC0415

    if hasattr(_trace, "_TRACER_PROVIDER_SET_ONCE"):
        _trace._TRACER_PROVIDER_SET_ONCE._done = False


@pytest.fixture(autouse=True)
def _ensure_valid_otel_provider() -> Generator[None, None, None]:
    """Ensure every test starts with a valid (non-recursive) TracerProvider.

    After the test, sets a fresh no-op TracerProvider to prevent
    recursion in the OTel proxy when the set-once guard is reset.
    """
    _reset_otel_provider_guard()
    yield
    _reset_otel_provider_guard()
    # Set a fresh SDK provider so the proxy doesn't recurse.
    from opentelemetry import trace as _trace  # noqa: PLC0415

    _trace.set_tracer_provider(TracerProvider())


@pytest.fixture
def otel_provider() -> tuple[TracerProvider, _InMemoryExporter]:
    """Configure a real TracerProvider with an in-memory exporter.

    Returns a (provider, exporter) tuple so tests can inspect exported
    spans.  Sets the global TracerProvider for the test duration.
    The autouse ``_ensure_valid_otel_provider`` fixture handles teardown.
    """
    from opentelemetry import trace as _trace  # noqa: PLC0415

    exporter = _InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    _reset_otel_provider_guard()
    _trace.set_tracer_provider(provider)
    return provider, exporter


class TestTracedSpanNullObject:
    """TracedSpan(None) must be a silent no-op for every method."""

    def test_none_span_set_attribute_is_noop(self) -> None:
        """set_attribute must not raise when the underlying span is None."""
        TracedSpan(None).set_attribute("key", "value")

    def test_none_span_record_exception_is_noop(self) -> None:
        """record_exception must not raise when the underlying span is None."""
        TracedSpan(None).record_exception(RuntimeError("test"))

    def test_none_span_set_status_is_noop(self) -> None:
        """set_status must not raise when the underlying span is None."""
        TracedSpan(None).set_status(StatusCode.ERROR, "something failed")


class TestTracedSpanDelegation:
    """TracedSpan delegates to the underlying span when it is not None."""

    def test_set_attribute_delegates_to_real_span(self) -> None:
        """set_attribute must forward key and value to the real span."""
        mock_span = MagicMock()
        TracedSpan(mock_span).set_attribute("stage.name", "VideoDownloader")
        mock_span.set_attribute.assert_called_once_with("stage.name", "VideoDownloader")

    def test_set_attributes_delegates_each_pair(self) -> None:
        """set_attributes must call set_attribute for each key-value pair."""
        mock_span = MagicMock()
        TracedSpan(mock_span).set_attributes({"k1": "v1", "k2": 42})
        assert mock_span.set_attribute.call_count == 2


class TestTracedSpanContextManager:
    """traced_span() context manager creates real OTel spans."""

    def test_traced_span_yields_traced_span_instance(self) -> None:
        """The yielded object must be a TracedSpan."""
        with traced_span("test.op") as span:
            assert isinstance(span, TracedSpan)

    def test_traced_span_creates_real_span_with_configured_provider(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """With a configured TracerProvider, a named span is exported."""
        _provider, exporter = otel_provider

        with traced_span("my.span"):
            pass

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "my.span" in span_names


class TestTracedDecorator:
    """@traced decorator wraps the function body in an OTel span."""

    def test_traced_preserves_return_value_and_creates_span(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """Decorated function returns its value; a span is exported."""
        _provider, exporter = otel_provider

        @traced("my.decorated_fn")
        def compute(x: int) -> int:
            return x * 2

        result = compute(21)
        assert result == 42

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "my.decorated_fn" in span_names

    @pytest.mark.asyncio
    async def test_traced_async_preserves_return_value_and_creates_span(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """Async decorated function returns its value; a span is exported."""
        _provider, exporter = otel_provider

        @traced("my.async_fn")
        async def async_compute(x: int) -> int:
            return x * 2

        result = await async_compute(21)
        assert result == 42

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "my.async_fn" in span_names

    @pytest.mark.asyncio
    async def test_traced_async_span_covers_full_coroutine(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """Async span must stay open until the coroutine completes.

        Verifies that child spans created inside the async function
        are correctly parented under the @traced span, not orphaned.
        """
        _provider, exporter = otel_provider

        @traced("parent.async_op")
        async def slow_op() -> str:
            with traced_span("child.inner_work"):
                await asyncio.sleep(0.01)
            return "done"

        result = await slow_op()
        assert result == "done"

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "parent.async_op" in span_names
        assert "child.inner_work" in span_names

        parent = next(s for s in spans if s.name == "parent.async_op")
        child = next(s for s in spans if s.name == "child.inner_work")

        # Child must be parented under the async span.
        assert child.parent is not None
        assert child.parent.span_id == parent.context.span_id

        # Parent span must have non-zero duration (covers the full await).
        assert parent.end_time is not None
        assert parent.start_time is not None
        assert parent.end_time > parent.start_time

    @pytest.mark.asyncio
    async def test_traced_async_propagates_exceptions(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """Exceptions in async functions must propagate through @traced."""
        _provider, exporter = otel_provider

        @traced("failing.async_fn")
        async def failing() -> None:
            msg = "async boom"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="async boom"):
            await failing()

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "failing.async_fn" in span_names

    @pytest.mark.asyncio
    async def test_traced_async_bound_method(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """@traced works on async bound methods (the production use case)."""
        _provider, exporter = otel_provider

        class Worker:
            @traced("Worker.run")
            async def run(self, x: int) -> int:
                return x + 1

        result = await Worker().run(41)
        assert result == 42

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "Worker.run" in span_names


class TestAsyncRunnerContextPropagation:
    """asyncio.Runner + contextvars.copy_context() span parenting.

    asyncio.Runner caches its context on the first run() call.
    Without passing context=contextvars.copy_context(), all
    subsequent run() calls reuse the first batch's OTel context,
    causing child spans to be mis-parented under batch 1's spans.
    """

    def test_runner_context_per_call_parents_correctly(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """Each Runner.run() call with copy_context() parents under its own batch span."""
        _provider, exporter = otel_provider
        runner = asyncio.Runner()

        async def create_child(name: str) -> None:
            with traced_span(name):
                pass

        with traced_span("batch_1"):
            runner.run(create_child("child_1"), context=contextvars.copy_context())
        with traced_span("batch_2"):
            runner.run(create_child("child_2"), context=contextvars.copy_context())

        runner.close()

        spans = exporter.get_finished_spans()
        child_1 = next(s for s in spans if s.name == "child_1")
        child_2 = next(s for s in spans if s.name == "child_2")
        batch_1 = next(s for s in spans if s.name == "batch_1")
        batch_2 = next(s for s in spans if s.name == "batch_2")

        assert child_1.parent is not None
        assert child_1.parent.span_id == batch_1.context.span_id
        assert child_2.parent is not None
        assert child_2.parent.span_id == batch_2.context.span_id

    def test_runner_without_context_copy_misparents(
        self,
        otel_provider: tuple[TracerProvider, _InMemoryExporter],
    ) -> None:
        """Without copy_context(), Runner reuses batch 1's context for batch 2.

        This test documents the Python 3.12 asyncio.Runner footgun that
        the context=contextvars.copy_context() fix addresses.
        """
        _provider, exporter = otel_provider
        runner = asyncio.Runner()

        async def create_child(name: str) -> None:
            with traced_span(name):
                pass

        with traced_span("batch_a"):
            runner.run(create_child("child_a"))
        with traced_span("batch_b"):
            runner.run(create_child("child_b"))

        runner.close()

        spans = exporter.get_finished_spans()
        child_b = next(s for s in spans if s.name == "child_b")
        batch_a = next(s for s in spans if s.name == "batch_a")

        # child_b should be under batch_b, but Runner reuses batch_a's
        # cached context -- so child_b is incorrectly under batch_a.
        assert child_b.parent is not None
        assert child_b.parent.span_id == batch_a.context.span_id


class TestArtifactNaming:
    """artifact_id and process_tag produce deterministic names."""

    def test_artifact_id_format(self) -> None:
        """artifact_id includes stage name, call_id, and process tag."""
        aid = artifact_id("VideoDownloader", "setup_1")
        assert aid.startswith("VideoDownloader_setup_1_")
        # Must contain the PID somewhere after the prefix.
        assert str(os.getpid()) in aid

    def test_process_tag_contains_pid(self) -> None:
        """process_tag includes the current OS PID."""
        tag = process_tag()
        assert str(os.getpid()) in tag


class TestGetOtlpEndpoint:
    """get_otlp_endpoint resolves the OTLP endpoint from env vars."""

    def test_default_when_no_env_vars(self) -> None:
        """Should return empty string when no env vars are set (OTLP disabled)."""
        from unittest.mock import patch  # noqa: PLC0415

        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            from cosmos_curator.core.utils.infra.tracing import get_otlp_endpoint  # noqa: PLC0415

            assert get_otlp_endpoint() == ""

    def test_traces_endpoint_takes_precedence(self) -> None:
        """OTEL_EXPORTER_OTLP_TRACES_ENDPOINT should win over general endpoint."""
        from unittest.mock import patch  # noqa: PLC0415

        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://general:4318",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://traces:4318",
            },
        ):
            from cosmos_curator.core.utils.infra.tracing import get_otlp_endpoint  # noqa: PLC0415

            assert get_otlp_endpoint() == "http://traces:4318"

    def test_general_endpoint_used_when_no_traces_endpoint(self) -> None:
        """OTEL_EXPORTER_OTLP_ENDPOINT should be used when traces-specific is not set."""
        from unittest.mock import patch  # noqa: PLC0415

        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://general:4318",
            },
            clear=True,
        ):
            from cosmos_curator.core.utils.infra.tracing import get_otlp_endpoint  # noqa: PLC0415

            assert get_otlp_endpoint() == "http://general:4318"


class TestTraceRootAnchor:
    """trace_root_anchor creates a root span with no parent."""

    def test_trace_root_anchor_yields_traced_span(self) -> None:
        """The yielded object must be a TracedSpan."""
        with trace_root_anchor("anchor") as span:
            assert isinstance(span, TracedSpan)
