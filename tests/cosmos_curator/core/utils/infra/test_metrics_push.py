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

"""Tests for ``cosmos_curator.core.utils.infra.metrics_push``.

All tests run hermetically: ``ray`` is mocked via
``monkeypatch.setitem(sys.modules, ...)``, ``requests.get`` is patched,
and the OTLP exporter is replaced with a ``MagicMock``.  No live Ray
cluster, no real HTTP, no GPU.
"""

import contextlib
import os
import re
import threading
import time
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from loguru import logger
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    Gauge,
    Histogram,
    MetricExportResult,
    Sum,
)

import cosmos_curator.core.utils.infra.metrics_push as _mp_module
from cosmos_curator.core.utils.infra.metrics_push import (
    DEFAULT_DROP_METRIC_NAMES_REGEX,
    ENV_OTLP_ENDPOINT,
    MetricsPushBackend,
    MetricsPushConfig,
    disable_metrics_push,
    enable_metrics_push,
    get_otlp_metrics_endpoint,
)

CANONICAL_PROM_PAYLOADS: dict[str, str] = {
    "gauge": (
        "# HELP ray_node_cpu_count CPU cores on this node.\n"
        "# TYPE ray_node_cpu_count gauge\n"
        'ray_node_cpu_count{NodeID="abc"} 8.0\n'
    ),
    "counter": (
        "# HELP ray_tasks_total Number of tasks executed.\n"
        "# TYPE ray_tasks_total counter\n"
        'ray_tasks_total{State="Finished"} 42.0\n'
        'ray_tasks_total{State="Failed"} 1.0\n'
    ),
    "histogram": (
        "# HELP ray_task_latency Task latency.\n"
        "# TYPE ray_task_latency histogram\n"
        'ray_task_latency_bucket{le="0.1"} 1\n'
        'ray_task_latency_bucket{le="0.5"} 3\n'
        'ray_task_latency_bucket{le="1.0"} 5\n'
        'ray_task_latency_bucket{le="+Inf"} 7\n'
        "ray_task_latency_count 7\n"
        "ray_task_latency_sum 2.5\n"
    ),
    "summary": (
        "# HELP ray_request_duration A summary.\n"
        "# TYPE ray_request_duration summary\n"
        'ray_request_duration{quantile="0.5"} 1.0\n'
        'ray_request_duration{quantile="0.9"} 2.0\n'
        "ray_request_duration_sum 30.0\n"
        "ray_request_duration_count 10.0\n"
    ),
}


@pytest.fixture(autouse=True)
def _reset_metrics_push_singleton() -> Generator[None]:
    """Snapshot and restore the module-level singleton across tests."""
    original = _mp_module._current_backend
    if original is not None:
        with contextlib.suppress(Exception):
            original.stop()
    _mp_module._current_backend = None
    yield
    if _mp_module._current_backend is not None:
        with contextlib.suppress(Exception):
            _mp_module._current_backend.stop()
    _mp_module._current_backend = original


@pytest.fixture(autouse=True)
def _clear_otlp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no stray OTel env vars bleed across tests."""
    for var in (
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "XENNA_RAY_METRICS_PORT",
        "COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES",
    ):
        monkeypatch.delenv(var, raising=False)


class TestEndpointResolution:
    """``MetricsPushConfig.resolved`` honours OTel env var precedence."""

    def test_cli_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit ``cli_otlp_endpoint`` (CLI flag) wins over both env vars.

        Mirrors :func:`enable_tracing` -- a CLI flag must never be
        silently overridden by a stale env var.
        """
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "https://metrics.env")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://generic.env")
        config = MetricsPushConfig(enabled=True).resolved(cli_otlp_endpoint="https://cli.example")
        assert config.otlp_endpoint == "https://cli.example"

    def test_metrics_specific_env_beats_generic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When CLI is empty, metrics-specific env beats the generic var."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "https://metrics.example")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otlp.example")
        config = MetricsPushConfig(enabled=True).resolved()
        assert config.otlp_endpoint == "https://metrics.example"

    def test_generic_env_used_when_metrics_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``OTEL_EXPORTER_OTLP_ENDPOINT`` is used when the metrics-specific var is empty."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otlp.example")
        config = MetricsPushConfig(enabled=True).resolved()
        assert config.otlp_endpoint == "https://otlp.example"

    def test_cli_used_when_env_unset(self) -> None:
        """With no OTel env vars, the explicit CLI endpoint is used."""
        config = MetricsPushConfig(enabled=True).resolved(cli_otlp_endpoint="https://cli.example")
        assert config.otlp_endpoint == "https://cli.example"

    def test_empty_when_nothing_set(self) -> None:
        """All sources empty -> empty endpoint (push will be a no-op)."""
        config = MetricsPushConfig(enabled=True).resolved()
        assert config.otlp_endpoint == ""

    def test_service_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``OTEL_SERVICE_NAME`` propagates into the config."""
        monkeypatch.setenv("OTEL_SERVICE_NAME", "my_custom_svc")
        config = MetricsPushConfig(enabled=True).resolved(cli_otlp_endpoint="https://x")
        assert config.service_name == "my_custom_svc"

    def test_get_otlp_metrics_endpoint_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``get_otlp_metrics_endpoint`` mirrors the resolver used internally."""
        assert get_otlp_metrics_endpoint() == ""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://generic")
        assert get_otlp_metrics_endpoint() == "https://generic"
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "https://specific")
        assert get_otlp_metrics_endpoint() == "https://specific"


class TestPromTextToOtlp:
    """Prom-text -> OTel ``MetricsData`` conversion."""

    NOW_NS = 1_700_000_000_000_000_000
    START_NS = 1_600_000_000_000_000_000  # arbitrary stable "first-seen" stamp

    @staticmethod
    def _start_time_for(_metric_name: str, _label_key: tuple[tuple[str, str], ...]) -> int:
        return TestPromTextToOtlp.START_NS

    @staticmethod
    def _convert(
        text: str,
        *,
        drop_pattern: "re.Pattern[str] | None" = None,
        series_labels: dict[str, str] | None = None,
    ) -> object:
        return _mp_module._prom_text_to_metrics_data(
            text,
            resource_attrs={"service.name": "test"},
            series_labels=series_labels or {},
            scope_name="test.scope",
            now_unix_nano=TestPromTextToOtlp.NOW_NS,
            start_time_for=TestPromTextToOtlp._start_time_for,
            drop_pattern=drop_pattern,
        )

    @staticmethod
    def _metrics(data: object) -> list:
        # MetricsData -> ResourceMetrics -> ScopeMetrics -> [Metric].
        return list(data.resource_metrics[0].scope_metrics[0].metrics)  # type: ignore[attr-defined]

    def test_empty_input_returns_none(self) -> None:
        """Empty prom text -> ``None`` so the caller skips export."""
        assert self._convert("") is None

    @pytest.mark.parametrize(
        ("payload_name", "expected_names"),
        [
            ("gauge", {"ray_node_cpu_count"}),
            ("counter", {"ray_tasks_total"}),
            ("histogram", {"ray_task_latency"}),
            ("summary", {"ray_request_duration", "ray_request_duration_sum", "ray_request_duration_count"}),
        ],
    )
    def test_canonical_payload_contracts(self, payload_name: str, expected_names: set[str]) -> None:
        """Canonical payload fixtures convert to expected metric families."""
        metrics = self._metrics(self._convert(CANONICAL_PROM_PAYLOADS[payload_name]))
        names = {metric.name for metric in metrics}
        assert names == expected_names

    def test_gauge_becomes_otel_gauge(self) -> None:
        """A simple gauge family maps to a single OTel ``Gauge`` metric."""
        text = CANONICAL_PROM_PAYLOADS["gauge"]
        metrics = self._metrics(self._convert(text))
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric.name == "ray_node_cpu_count"
        assert isinstance(metric.data, Gauge)
        assert len(metric.data.data_points) == 1
        point = metric.data.data_points[0]
        assert point.value == 8.0
        assert point.attributes == {"NodeID": "abc"}
        assert point.time_unix_nano == self.NOW_NS

    def test_counter_preserves_total_suffix_and_is_monotonic(self) -> None:
        """A counter family keeps its Prom name as monotonic cumulative ``Sum``."""
        text = CANONICAL_PROM_PAYLOADS["counter"]
        metrics = self._metrics(self._convert(text))
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric.name == "ray_tasks_total"
        assert isinstance(metric.data, Sum)
        assert metric.data.is_monotonic is True
        assert metric.data.aggregation_temporality == AggregationTemporality.CUMULATIVE
        values = {tuple(sorted(p.attributes.items())): p.value for p in metric.data.data_points}
        assert values == {
            (("State", "Finished"),): 42.0,
            (("State", "Failed"),): 1.0,
        }

    def test_histogram_buckets_become_per_bucket_counts(self) -> None:
        """Cumulative buckets convert to OTel per-bucket counts; +Inf is dropped from bounds."""
        text = CANONICAL_PROM_PAYLOADS["histogram"]
        metrics = self._metrics(self._convert(text))
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric.name == "ray_task_latency"
        assert isinstance(metric.data, Histogram)
        point = metric.data.data_points[0]
        assert point.count == 7
        assert point.sum == 2.5
        # +Inf is dropped from explicit_bounds; cumulative -> per-bucket.
        assert list(point.explicit_bounds) == [0.1, 0.5, 1.0]
        assert list(point.bucket_counts) == [1, 2, 2, 2]
        assert metric.data.aggregation_temporality == AggregationTemporality.CUMULATIVE

    def test_summary_emits_sample_named_gauges(self) -> None:
        """Summary samples preserve their own metric names (base + _sum + _count)."""
        text = CANONICAL_PROM_PAYLOADS["summary"]
        metrics = self._metrics(self._convert(text))
        names = {metric.name for metric in metrics}
        assert names == {"ray_request_duration", "ray_request_duration_sum", "ray_request_duration_count"}
        quantile_metric = next(m for m in metrics if m.name == "ray_request_duration")
        assert isinstance(quantile_metric.data, Gauge)
        assert len(quantile_metric.data.data_points) == 2

    def test_resource_attrs_attached(self) -> None:
        """Resource attrs stay on the OTel resource (typically ``service.name`` only)."""
        text = "# TYPE g gauge\ng 1.0\n"
        data = _mp_module._prom_text_to_metrics_data(
            text,
            resource_attrs={"service.name": "svc"},
            series_labels={"host.name": "host01"},
            scope_name="cosmos_curator.ray_metrics",
            now_unix_nano=self.NOW_NS,
            start_time_for=TestPromTextToOtlp._start_time_for,
        )
        assert data is not None
        resource = data.resource_metrics[0].resource  # type: ignore[attr-defined]
        assert resource.attributes["service.name"] == "svc"
        point = data.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points[0]  # type: ignore[attr-defined]
        assert point.attributes["host.name"] == "host01"

    def test_series_labels_on_data_points(self) -> None:
        """Run/node labels are copied onto every exported data point."""
        text = '# TYPE ray_cluster_active_nodes gauge\nray_cluster_active_nodes{node_type="head"} 1.0\n'
        data = self._convert(
            text,
            series_labels={"slurm_job_id": "100", "ray.node_id": "node-a"},
        )
        point = self._metrics(data)[0].data.data_points[0]
        assert point.attributes["slurm_job_id"] == "100"
        assert point.attributes["ray.node_id"] == "node-a"
        assert point.attributes["node_type"] == "head"

    def test_counter_uses_stable_start_time(self) -> None:
        """Counter data points carry the per-series start_time, not the tick time."""
        text = '# TYPE ray_tasks_total counter\nray_tasks_total{State="Finished"} 42.0\n'
        metrics = self._metrics(self._convert(text))
        point = metrics[0].data.data_points[0]
        assert point.start_time_unix_nano == self.START_NS
        assert point.time_unix_nano == self.NOW_NS
        assert point.start_time_unix_nano != point.time_unix_nano

    def test_histogram_uses_stable_start_time(self) -> None:
        """Histogram data points carry the per-series start_time."""
        text = (
            "# TYPE ray_task_latency histogram\n"
            'ray_task_latency_bucket{le="1.0"} 1\n'
            'ray_task_latency_bucket{le="+Inf"} 1\n'
            "ray_task_latency_count 1\n"
            "ray_task_latency_sum 0.5\n"
        )
        metrics = self._metrics(self._convert(text))
        point = metrics[0].data.data_points[0]
        assert point.start_time_unix_nano == self.START_NS
        assert point.time_unix_nano == self.NOW_NS

    def test_gauge_keeps_tick_time_as_start(self) -> None:
        """Gauges are point-in-time, so start_time stays at tick time (spec)."""
        text = "# TYPE g gauge\ng 1.0\n"
        metrics = self._metrics(self._convert(text))
        point = metrics[0].data.data_points[0]
        assert point.start_time_unix_nano == self.NOW_NS
        assert point.time_unix_nano == self.NOW_NS

    def test_drop_pattern_filters_matching_metrics(self) -> None:
        """A compiled drop pattern removes matching metrics from the output."""
        text = (
            "# TYPE ray_tasks_total counter\nray_tasks_total 1.0\n"
            "# TYPE ray_scheduler_inflight_decisions_total counter\nray_scheduler_inflight_decisions_total 5.0\n"
            "# TYPE ray_node_cpu_count gauge\nray_node_cpu_count 8.0\n"
        )
        pattern = re.compile(r"^(ray_tasks|ray_scheduler_.*)$")
        data = self._convert(text, drop_pattern=pattern)
        names = [m.name for m in self._metrics(data)]
        assert names == ["ray_node_cpu_count"]

    def test_drop_pattern_none_keeps_everything(self) -> None:
        """drop_pattern=None preserves all metrics (the no-filter path)."""
        text = (
            "# TYPE ray_tasks_total counter\nray_tasks_total 1.0\n"
            "# TYPE ray_node_cpu_count gauge\nray_node_cpu_count 8.0\n"
        )
        data = self._convert(text, drop_pattern=None)
        names = {m.name for m in self._metrics(data)}
        assert names == {"ray_tasks_total", "ray_node_cpu_count"}


class TestSeriesLabels:
    """``MetricsPushBackend._series_labels_for`` run-metadata attachment."""

    def test_includes_slurm_attrs_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Slurm env vars appear as series labels when the toggle is on."""
        monkeypatch.setenv("SLURM_JOB_ID", "100")
        monkeypatch.setenv("SLURM_JOB_USER", "alice")
        config = MetricsPushConfig(enabled=True, otlp_endpoint="http://localhost:4318")
        backend = MetricsPushBackend(config)
        labels = backend._series_labels_for("node-a", "http://10.0.0.1:9002/metrics", "node03.cluster")
        assert labels["host.name"] == "node03"
        assert "instance" not in labels
        assert labels["scrape_pid"] == str(os.getpid())
        assert labels["scrape_tid"] == str(threading.get_ident())
        assert labels["scrape_native_tid"] == str(threading.get_native_id())
        assert labels["slurm_job_id"] == "100"
        assert labels["slurm_job_user"] == "alice"

    def test_omits_run_attrs_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``include_run_attributes=False`` skips Slurm metadata on series labels."""
        monkeypatch.setenv("SLURM_JOB_ID", "100")
        config = MetricsPushConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4318",
            include_run_attributes=False,
        )
        backend = MetricsPushBackend(config)
        labels = backend._series_labels_for("node-a", "http://10.0.0.1:9002/metrics", "node01")
        assert "slurm_job_id" not in labels
        assert "instance" not in labels
        assert labels["scrape_pid"] == str(os.getpid())
        assert labels["scrape_tid"] == str(threading.get_ident())
        assert labels["scrape_native_tid"] == str(threading.get_native_id())
        assert labels["host.name"] == "node01"

    def test_instance_label_fallback_when_url_unexpected(self) -> None:
        """Malformed scrape URLs still produce a stable instance-id value."""
        config = MetricsPushConfig(enabled=True, otlp_endpoint="http://localhost:4318")
        backend = MetricsPushBackend(config)
        assert backend._instance_label_for("not-a-url", node_host="node01", node_id="node-a") == "node01"

    def test_resource_carries_service_name_only(self) -> None:
        """Resource attrs are minimal so Mimir maps ``service.name`` to ``job``."""
        config = MetricsPushConfig(enabled=True, otlp_endpoint="http://localhost:4318", service_name="svc")
        backend = MetricsPushBackend(config)
        assert backend._resource_attrs_for() == {"service.name": "svc"}

    def test_resource_can_carry_service_instance_id(self) -> None:
        """When provided, ``service.instance.id`` is set on resource attrs."""
        config = MetricsPushConfig(enabled=True, otlp_endpoint="http://localhost:4318", service_name="svc")
        backend = MetricsPushBackend(config)
        assert backend._resource_attrs_for(service_instance_id="node01:9002") == {
            "service.name": "svc",
            "service.instance.id": "node01:9002",
        }


class TestEnumerateTargets:
    """``MetricsPushBackend._enumerate_targets`` filters by alive + skips on missing port."""

    def _backend(self) -> MetricsPushBackend:
        config = MetricsPushConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4318",
        )
        return MetricsPushBackend(config)

    def test_alive_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dead nodes are skipped; alive ones yield (node_id, url) pairs."""
        fake_nodes = [
            {
                "Alive": True,
                "NodeID": "node-a",
                "NodeManagerAddress": "10.0.0.1",
                "MetricsExportPort": 9100,
            },
            {
                "Alive": False,
                "NodeID": "node-b",
                "NodeManagerAddress": "10.0.0.2",
                "MetricsExportPort": 9100,
            },
        ]
        ray_module = MagicMock()
        ray_module.nodes.return_value = fake_nodes
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        backend = self._backend()
        targets = backend._enumerate_targets()
        assert targets == [("node-a", "http://10.0.0.1:9100/metrics", "10.0.0.1")]

    def test_prefers_hostname_for_label_when_both_fields_exist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use NodeManagerHostname for labels while scraping NodeManagerAddress."""
        fake_nodes = [
            {
                "Alive": True,
                "NodeID": "node-a",
                "NodeManagerAddress": "10.0.0.1",
                "NodeManagerHostname": "worker-0.cluster.local",
                "MetricsExportPort": 9100,
            },
        ]
        ray_module = MagicMock()
        ray_module.nodes.return_value = fake_nodes
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        backend = self._backend()
        targets = backend._enumerate_targets()
        assert targets == [("node-a", "http://10.0.0.1:9100/metrics", "worker-0.cluster.local")]

    def test_skips_when_port_missing_or_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nodes with MetricsExportPort=0 or missing are skipped (no fallback)."""
        fake_nodes = [
            {
                "Alive": True,
                "NodeID": "node-zero",
                "NodeManagerAddress": "10.0.0.1",
                "MetricsExportPort": 0,
            },
            {
                "Alive": True,
                "NodeID": "node-missing",
                "NodeManagerAddress": "10.0.0.2",
            },
            {
                "Alive": True,
                "NodeID": "node-good",
                "NodeManagerAddress": "10.0.0.3",
                "MetricsExportPort": 9002,
            },
        ]
        ray_module = MagicMock()
        ray_module.nodes.return_value = fake_nodes
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        backend = self._backend()
        targets = backend._enumerate_targets()
        assert targets == [("node-good", "http://10.0.0.3:9002/metrics", "10.0.0.3")]

    def test_ray_raise_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raising ray.nodes() must not propagate; return empty target list."""
        ray_module = MagicMock()
        ray_module.nodes.side_effect = RuntimeError("gcs unavailable")
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        backend = self._backend()
        assert backend._enumerate_targets() == []


class TestResilientMetricsExporter:
    """``_ResilientMetricsExporter`` suppresses repeated connection failures."""

    @staticmethod
    def _make_delegate(*, side_effect: object = None) -> MagicMock:
        mock = MagicMock()
        mock.export.return_value = MetricExportResult.SUCCESS
        if side_effect is not None:
            mock.export.side_effect = side_effect
        return mock

    def test_success_path_resets_counter(self) -> None:
        """Successful exports keep the wrapper enabled and reset the failure counter."""
        delegate = self._make_delegate()
        wrapper = _mp_module._ResilientMetricsExporter(delegate)
        assert wrapper.export(object()) is True
        assert wrapper.disabled is False

    def test_disables_after_threshold_consecutive_connection_errors(self) -> None:
        """Three consecutive ConnectionErrors trip the disable threshold."""
        delegate = self._make_delegate(side_effect=ConnectionError("refused"))
        wrapper = _mp_module._ResilientMetricsExporter(delegate)

        for _ in range(3):
            assert wrapper.export(object()) is False
        assert wrapper.disabled is True

        # Further calls become silent no-ops; delegate is not invoked.
        delegate.export.reset_mock()
        assert wrapper.export(object()) is True
        delegate.export.assert_not_called()

    def test_reraises_non_connection_error(self) -> None:
        """Non-connection errors propagate."""
        delegate = self._make_delegate(side_effect=ValueError("bad data"))
        wrapper = _mp_module._ResilientMetricsExporter(delegate)
        with pytest.raises(ValueError, match="bad data"):
            wrapper.export(object())
        assert wrapper.disabled is False

    def test_failure_response_increments_counter(self) -> None:
        """A FAILURE return value also counts toward the threshold."""
        delegate = self._make_delegate()
        delegate.export.return_value = MetricExportResult.FAILURE
        wrapper = _mp_module._ResilientMetricsExporter(delegate)

        for _ in range(3):
            wrapper.export(object())
        assert wrapper.disabled is True

    def test_push_failures_log_at_warning(self) -> None:
        """OTLP export failures must be visible at WARNING, not only DEBUG."""
        messages: list[str] = []
        sink_id = logger.add(
            lambda msg: messages.append(msg.record["message"]),
            format="{message}",
            level="WARNING",
        )
        try:
            delegate = self._make_delegate(side_effect=ConnectionError("refused"))
            wrapper = _mp_module._ResilientMetricsExporter(delegate)

            wrapper.export(object())
            assert any("[otel-metrics] OTLP metrics push failed (1/3)" in msg for msg in messages)

            messages.clear()
            for _ in range(2):
                wrapper.export(object())
            assert wrapper.disabled is True
            assert any("Disabling remote OTLP metrics push after 3 consecutive failures" in msg for msg in messages)
        finally:
            logger.remove(sink_id)

    def test_shutdown_delegates(self) -> None:
        """``shutdown`` forwards to the delegate without raising."""
        delegate = self._make_delegate()
        wrapper = _mp_module._ResilientMetricsExporter(delegate)
        wrapper.shutdown()
        delegate.shutdown.assert_called_once()


class TestBackendLifecycle:
    """``MetricsPushBackend.start`` / ``stop`` / ``flush`` thread management."""

    def _build_backend(self, monkeypatch: pytest.MonkeyPatch) -> MetricsPushBackend:
        """Build a backend with mocked Ray nodes and a fast-tick interval."""
        fake_nodes = [
            {
                "Alive": True,
                "NodeID": "node-a",
                "NodeManagerAddress": "127.0.0.1",
                "MetricsExportPort": 9002,
            },
        ]
        ray_module = MagicMock()
        ray_module.nodes.return_value = fake_nodes
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        # Patch requests.get to return a trivial gauge metric body.
        prom_body = "# TYPE ray_demo gauge\nray_demo 1.0\n"
        response = MagicMock()
        response.ok = True
        response.text = prom_body
        response.status_code = 200
        monkeypatch.setattr(_mp_module.requests, "get", MagicMock(return_value=response))

        config = MetricsPushConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4318",
            interval_seconds=1,
            scrape_timeout_seconds=1,
        )
        backend = MetricsPushBackend(config)

        # Replace OTLPMetricExporter with a mock so no real network IO happens.
        captured: list[MagicMock] = []

        class _FakeExporterClass:
            def __init__(self) -> None:
                self.instance = MagicMock()
                self.instance.export.return_value = MetricExportResult.SUCCESS
                captured.append(self.instance)

            def __call__(self, **_: object) -> MagicMock:
                return self.instance

        fake_factory = _FakeExporterClass()
        import opentelemetry.exporter.otlp.proto.http.metric_exporter as exporter_module  # noqa: PLC0415

        monkeypatch.setattr(exporter_module, "OTLPMetricExporter", fake_factory)
        backend._captured_exporter = captured  # type: ignore[attr-defined]
        return backend

    def test_start_runs_one_tick_and_stop_is_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single scrape happens shortly after ``start``; ``stop`` joins quickly."""
        backend = self._build_backend(monkeypatch)
        tick_seen = threading.Event()
        original_tick = backend._scrape_and_push_once

        def _wrapped_tick() -> None:
            original_tick()
            if backend.tick_count > 0:
                tick_seen.set()

        monkeypatch.setattr(backend, "_scrape_and_push_once", _wrapped_tick)
        backend.start()

        assert tick_seen.wait(timeout=2.0)
        assert backend.tick_count >= 1

        start_stop = time.monotonic()
        backend.stop(join_timeout=2.0)
        assert time.monotonic() - start_stop < 2.0

    def test_skipped_when_disabled(self) -> None:
        """``start`` is a no-op when ``enabled=False`` -- no thread."""
        config = MetricsPushConfig(enabled=False, otlp_endpoint="http://localhost:4318")
        backend = MetricsPushBackend(config)
        backend.start()
        assert backend._thread is None  # type: ignore[attr-defined]

    def test_skipped_when_endpoint_empty(self) -> None:
        """``start`` is a no-op when no endpoint is configured."""
        config = MetricsPushConfig(enabled=True, otlp_endpoint="")
        backend = MetricsPushBackend(config)
        backend.start()
        assert backend._thread is None  # type: ignore[attr-defined]


class TestEnableMetricsPushModuleApi:
    """``enable_metrics_push`` / ``disable_metrics_push`` module helpers."""

    def test_enable_does_not_touch_ray_port_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``enable_metrics_push`` must NOT set ``XENNA_RAY_METRICS_PORT``.

        Setting the env var here would cause ``ray.init`` callers to
        pin the Prometheus port in local mode, which can collide
        with other listeners on the host.  The driver-side scraper
        discovers each node's port via ``ray.nodes()`` instead, so
        no curator-side pinning is needed (or wanted).
        """
        import opentelemetry.exporter.otlp.proto.http.metric_exporter as exporter_module  # noqa: PLC0415

        monkeypatch.setattr(exporter_module, "OTLPMetricExporter", lambda **_: MagicMock())

        ray_module = MagicMock()
        ray_module.nodes.side_effect = RuntimeError("not running")
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        # Sanity: clean slate.
        assert __import__("os").environ.get("XENNA_RAY_METRICS_PORT") is None
        enable_metrics_push(
            MetricsPushConfig(enabled=True),
            cli_otlp_endpoint="http://localhost:4318",
        )
        try:
            assert __import__("os").environ.get("XENNA_RAY_METRICS_PORT") is None
        finally:
            disable_metrics_push(join_timeout=1.0)

    def test_enable_preserves_existing_ray_port_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pre-existing ``XENNA_RAY_METRICS_PORT`` (Slurm/NVCF bootstrap, manual override) is not touched."""
        import opentelemetry.exporter.otlp.proto.http.metric_exporter as exporter_module  # noqa: PLC0415

        monkeypatch.setattr(exporter_module, "OTLPMetricExporter", lambda **_: MagicMock())

        ray_module = MagicMock()
        ray_module.nodes.side_effect = RuntimeError("not running")
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        monkeypatch.setenv("XENNA_RAY_METRICS_PORT", "9123")
        enable_metrics_push(
            MetricsPushConfig(enabled=True),
            cli_otlp_endpoint="http://localhost:4318",
        )
        try:
            assert __import__("os").environ.get("XENNA_RAY_METRICS_PORT") == "9123"
        finally:
            disable_metrics_push(join_timeout=1.0)

    def test_noop_without_endpoint(self) -> None:
        """No endpoint -> no singleton installed."""
        enable_metrics_push(MetricsPushConfig(enabled=True))
        assert _mp_module._current_backend is None

    def test_noop_when_disabled(self) -> None:
        """``enabled=False`` -> no singleton installed."""
        enable_metrics_push(
            MetricsPushConfig(enabled=False),
            cli_otlp_endpoint="http://localhost:4318",
        )
        assert _mp_module._current_backend is None

    def test_disable_is_idempotent(self) -> None:
        """Calling ``disable_metrics_push`` without an active backend is safe."""
        disable_metrics_push()
        disable_metrics_push()

    def test_cli_endpoint_written_to_metrics_env_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI metrics endpoint updates only metrics-specific OTLP env var."""
        import opentelemetry.exporter.otlp.proto.http.metric_exporter as exporter_module  # noqa: PLC0415

        monkeypatch.setattr(exporter_module, "OTLPMetricExporter", lambda **_: MagicMock())
        ray_module = MagicMock()
        ray_module.nodes.side_effect = RuntimeError("not running")
        monkeypatch.setitem(__import__("sys").modules, "ray", ray_module)

        # Simulate tracing pinning a generic endpoint that must remain untouched.
        monkeypatch.setenv(ENV_OTLP_ENDPOINT, "https://stale.env")
        enable_metrics_push(
            MetricsPushConfig(enabled=True),
            cli_otlp_endpoint="https://cli.example",
        )
        try:
            assert _mp_module._current_backend is not None
            assert _mp_module._current_backend._config.otlp_endpoint == "https://cli.example"
            assert __import__("os").environ.get(ENV_OTLP_ENDPOINT) == "https://stale.env"
            assert (
                __import__("os").environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") == "https://cli.example/v1/metrics"
            )
        finally:
            disable_metrics_push(join_timeout=1.0)


class TestDropRegex:
    """Drop-regex compilation and the helm-matching default."""

    def test_default_matches_helm_drop_list(self) -> None:
        """The default regex drops the same name patterns as the helm chart.

        Patterns mirror ``charts/cosmos-curator/values.yaml`` metrics
        relabel ``drop`` rules; if you change one, change both.
        """
        pattern = re.compile(DEFAULT_DROP_METRIC_NAMES_REGEX)
        should_drop = [
            "ray_tasks",
            "ray_actors",
            "ray_object_store_memory",
            "ray_object_store_dist",
            "ray_total_lineage_bytes",
            "ray_gcs_placement_count",
            "ray_gcs_storage_bytes",
            "ray_scheduler_inflight_decisions",
            "ray_pull_manager_active",
        ]
        should_keep = [
            "ray_node_cpu_count",
            "ray_node_memory_bytes",
            "ray_object_store_pending",
            "custom_curator_metric",
        ]
        for name in should_drop:
            assert pattern.fullmatch(name), f"expected drop: {name}"
        for name in should_keep:
            assert pattern.fullmatch(name) is None, f"expected keep: {name}"

    def test_empty_regex_disables_filtering(self) -> None:
        """drop_regex='' compiles to None (no filtering)."""
        config = MetricsPushConfig(enabled=True, otlp_endpoint="http://x", drop_regex="")
        assert config.compiled_drop_pattern() is None

    def test_non_empty_regex_compiles(self) -> None:
        """Non-empty drop_regex returns a compiled ``re.Pattern``."""
        config = MetricsPushConfig(enabled=True, otlp_endpoint="http://x", drop_regex=r"^foo$")
        pattern = config.compiled_drop_pattern()
        assert pattern is not None
        assert pattern.fullmatch("foo")
        assert pattern.fullmatch("bar") is None


class TestStartTimeTracker:
    """``MetricsPushBackend`` keeps a stable start_time_unix_nano per series."""

    def _backend(self) -> MetricsPushBackend:
        config = MetricsPushConfig(enabled=True, otlp_endpoint="http://localhost:4318", drop_regex="")
        return MetricsPushBackend(config)

    def test_first_call_records_now(self) -> None:
        """First lookup writes the current tick time."""
        backend = self._backend()
        resolver = backend._make_start_time_for(node_id="n1", now_unix_nano=100)
        assert resolver("ray_tasks", ()) == 100
        assert backend._series_start_times[("n1", "ray_tasks", ())] == 100

    def test_subsequent_call_returns_first_seen(self) -> None:
        """The original first-seen timestamp is returned on subsequent ticks."""
        backend = self._backend()
        first = backend._make_start_time_for(node_id="n1", now_unix_nano=100)
        assert first("ray_tasks", ()) == 100
        # Later "tick" -- now is 200 but the same series must keep 100.
        second = backend._make_start_time_for(node_id="n1", now_unix_nano=200)
        assert second("ray_tasks", ()) == 100

    def test_distinct_series_get_distinct_timestamps(self) -> None:
        """Different (node_id, name, labels) keys track independently."""
        backend = self._backend()
        r100 = backend._make_start_time_for(node_id="n1", now_unix_nano=100)
        r200 = backend._make_start_time_for(node_id="n2", now_unix_nano=200)
        # n1's series locks in at 100.
        assert r100("ray_tasks", (("State", "Finished"),)) == 100
        # n2's same-name series locks in at 200 (different node).
        assert r200("ray_tasks", (("State", "Finished"),)) == 200
        # Same node, different labels = different series.
        r300 = backend._make_start_time_for(node_id="n1", now_unix_nano=300)
        assert r300("ray_tasks", (("State", "Failed"),)) == 300
        # Original n1/Finished still 100.
        assert r300("ray_tasks", (("State", "Finished"),)) == 100


def test_no_dangling_thread_after_test_run() -> None:
    """Sanity check: this test suite leaves no metrics-push thread behind."""
    names = {t.name for t in threading.enumerate()}
    assert "cosmos-curator-metrics-push" not in names


class TestOtlpPushScope:
    """otlp_push_scope: the env-driven run boundary used by run-pipeline."""

    @staticmethod
    def _record(monkeypatch: pytest.MonkeyPatch, calls: list[str], *, started: bool) -> None:
        def fake_enable(config: object) -> bool:
            del config
            calls.append("enable")
            return started

        monkeypatch.setattr(_mp_module, "enable_metrics_push", fake_enable)
        monkeypatch.setattr(_mp_module, "flush_metrics_push", lambda: calls.append("flush"))
        monkeypatch.setattr(_mp_module, "disable_metrics_push", lambda: calls.append("disable"))

    def test_stop_happens_after_the_body_when_the_scope_started_the_push(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enable -> body -> flush -> disable."""
        calls: list[str] = []
        self._record(monkeypatch, calls, started=True)
        with _mp_module.otlp_push_scope():
            calls.append("run")
        assert calls == ["enable", "run", "flush", "disable"]

    def test_does_not_stop_a_backend_it_did_not_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend already active (profiling scope, fixture) is left running."""
        calls: list[str] = []
        self._record(monkeypatch, calls, started=False)
        with _mp_module.otlp_push_scope():
            pass
        assert calls == ["enable"]

    def test_body_still_runs_when_enable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed enable warns; the body runs and nothing is stopped."""
        calls: list[str] = []
        self._record(monkeypatch, calls, started=True)

        def boom(config: object) -> bool:
            del config
            msg = "otel exploded {'nested braces'}"
            raise RuntimeError(msg)

        monkeypatch.setattr(_mp_module, "enable_metrics_push", boom)
        with _mp_module.otlp_push_scope():
            calls.append("run")
        assert calls == ["run"]

    def test_run_attributes_env_opt_out_reaches_the_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES=0 is forwarded, not overridden to True."""
        monkeypatch.setenv("COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES", "0")
        calls: list[str] = []
        monkeypatch.setattr(_mp_module, "flush_metrics_push", lambda: calls.append("flush"))
        monkeypatch.setattr(_mp_module, "disable_metrics_push", lambda: calls.append("disable"))
        seen: list[bool] = []

        def capture(config: _mp_module.MetricsPushConfig) -> bool:
            seen.append(config.include_run_attributes)
            return True

        monkeypatch.setattr(_mp_module, "enable_metrics_push", capture)
        with _mp_module.otlp_push_scope():
            pass
        assert seen == [False]

    @pytest.mark.parametrize(
        ("interval_env", "expected_interval"),
        [("45", 45), ("abc", _mp_module.DEFAULT_INTERVAL_SECONDS), ("0", _mp_module.DEFAULT_INTERVAL_SECONDS)],
    )
    def test_env_feeds_the_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        interval_env: str,
        expected_interval: int,
    ) -> None:
        """Interval and drop regex come from env; an invalid interval falls back to the default."""
        captured: list[MetricsPushConfig] = []

        def fake_enable(config: MetricsPushConfig) -> bool:
            captured.append(config)
            return False

        monkeypatch.setattr(_mp_module, "enable_metrics_push", fake_enable)
        monkeypatch.setenv(_mp_module.OTLP_METRICS_PUSH_INTERVAL_ENV_VAR, interval_env)
        monkeypatch.setenv(_mp_module._ENV_DROP_REGEX, "custom_.*")
        with _mp_module.otlp_push_scope():
            pass
        (config,) = captured
        assert config.interval_seconds == expected_interval
        assert config.drop_regex == "custom_.*"
