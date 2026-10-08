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

"""Driver-side periodic metrics scraper and OTLP HTTP pusher.

Mirrors the structure of :mod:`tracing_hook` but for **metrics**:

* :class:`MetricsPushConfig` -- frozen attrs configuration with
  defaults and environment-variable resolution.
* :class:`MetricsPushBackend` -- owns the background daemon thread
  that scrapes every Ray node's Prometheus ``/metrics`` endpoint
  (via :func:`ray.nodes`) and pushes the data through
  :class:`OTLPMetricExporter` (HTTP/protobuf, path ``/v1/metrics``)
  on a configurable interval.
* :func:`enable_metrics_push` / :func:`disable_metrics_push` /
  :func:`flush_metrics_push` -- module-level lifecycle helpers
  matching the trace shape.

Design choices (see also the project plan):

* **Driver-only**: a single background thread on the pipeline driver
  enumerates Ray nodes and scrapes each one over HTTP.  No per-node
  Ray actor is created; lifecycle stays tied to the driver process.
* **OTLP HTTP**: we reuse the already-installed
  ``opentelemetry-exporter-otlp-proto-http`` package.  Endpoint
  resolution mirrors traces (CLI flag -> ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT``
  -> ``OTEL_EXPORTER_OTLP_ENDPOINT``).
* **Source = Prometheus text**: we scrape Ray's existing Prometheus
  endpoint and translate, so custom metrics emitted by stages are
  preserved (Ray's native OTel metrics mode does not include them).
* **Resilient**: a wrapper analogous to ``_ResilientOtlpExporter``
  disables remote push after N consecutive connection failures and
  the scrape loop swallows per-node and per-tick errors.  Metrics
  must never crash the pipeline.

Configuration is opt-in.  When ``MetricsPushConfig.enabled`` is
``False`` (the default), :func:`enable_metrics_push` is a no-op.
"""

import errno
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from urllib.parse import urlsplit

import attrs
import requests
from loguru import logger
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    Gauge,
    Histogram,
    HistogramDataPoint,
    Metric,
    MetricsData,
    NumberDataPoint,
    ResourceMetrics,
    ScopeMetrics,
    Sum,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from prometheus_client.parser import text_string_to_metric_families

from cosmos_curator.core.utils.environment import OTLP_METRICS_PUSH_INTERVAL_ENV_VAR, env_int
from cosmos_curator.core.utils.infra.run_attributes import (
    collect_run_attributes,
    otlp_run_attributes_enabled,
    set_otlp_run_attributes_enabled,
    short_host_label,
)
from cosmos_curator.core.utils.infra.tracing import ENV_OTLP_ENDPOINT

# Standard OTel env var for the metrics-specific endpoint.  Takes
# precedence over OTEL_EXPORTER_OTLP_ENDPOINT in the OTLP HTTP
# metric exporter (see opentelemetry-exporter-otlp-proto-http).
_ENV_OTLP_METRICS_ENDPOINT = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"

# Standard OTel env var for the resource service name.  Same value
# used by the tracing backend; kept in sync so traces and metrics
# share a service.name in the backend.
_ENV_SERVICE_NAME = "OTEL_SERVICE_NAME"

# Env override for the drop regex used by otlp_push_scope; "" disables filtering.
_ENV_DROP_REGEX = "COSMOS_CURATOR_OTLP_METRICS_PUSH_DROP_REGEX"

# Default scrape interval and per-node HTTP timeout (seconds).
DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_SCRAPE_TIMEOUT_SECONDS = 5

# Maximum consecutive push failures before the resilient exporter
# disables itself for the rest of the process lifetime.
_MAX_CONSECUTIVE_PUSH_FAILURES = 3


def _ensure_metrics_path(endpoint: str) -> str:
    """Return *endpoint* with ``/v1/metrics`` suffix exactly once."""
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/v1/metrics"):
        return trimmed
    return f"{trimmed}/v1/metrics"


def _log_otlp_push_failure(*, failures: int, detail: str) -> None:
    """Emit a visible log line for an OTLP metrics export failure."""
    if failures >= _MAX_CONSECUTIVE_PUSH_FAILURES:
        logger.warning(
            f"[otel-metrics] Disabling remote OTLP metrics push after {failures} consecutive failures: "
            f"{detail}. No further metrics will be sent for the rest of this run.",
        )
        return
    logger.warning(
        f"[otel-metrics] OTLP metrics push failed ({failures}/{_MAX_CONSECUTIVE_PUSH_FAILURES}): {detail}",
    )


# Per-series first-observation timestamp resolver.  Given a metric
# name (after ``_total`` stripping) and a stable label-key, returns
# the Unix nanosecond timestamp to use as ``start_time_unix_nano``.
# The backend implementation memoises first-seen times so cumulative
# counters and histograms keep a stable start across pushes (OTLP
# backends interpret a moving ``start_time_unix_nano`` as a counter
# reset).
SeriesStartTimeFn = Callable[[str, tuple[tuple[str, str], ...]], int]


# PUSH-SIDE drop list applied by the driver-side metrics scraper
# (this module) before handing metrics to the OTLP exporter.
# Mirrors the SCRAPE-SIDE ``metric_relabel_configs`` drop list at:
#   charts/cosmos-curator/values.yaml
#   -> .metrics.prometheus.scrapeConfigs[0].metric_relabel_configs
# so chart-scraped (helm) and driver-pushed (local / Slurm / NVCF)
# exports produce the same series set for the same Ray cluster.
# Keep both lists in sync when the Ray-side metric inventory
# changes.  Note: chart regexes are anchored to the prom sample
# name (e.g. `_bucket` suffix); these are anchored to the OTel
# metric name (after `_total` is stripped from counters and
# histograms are merged to their family name).  The two encode the
# same intent in their respective domains -- e.g. chart's
# ``ray_object_store_dist_bucket`` corresponds to ``ray_object_store_dist``
# here because we filter at the histogram family level rather than
# at the bucket sample level.  Empty string disables filtering
# entirely.
DEFAULT_DROP_METRIC_NAMES_REGEX = (
    r"^("
    r"ray_tasks"
    r"|ray_actors"
    r"|ray_object_store_memory"
    r"|ray_object_store_dist"
    r"|ray_total_lineage_bytes"
    r"|ray_gcs_placement_.*"
    r"|ray_gcs_storage_.*"
    r"|ray_scheduler_.*"
    r"|ray_pull_manager_.*"
    r")$"
)


def get_otlp_metrics_endpoint() -> str:
    """Resolve the OTLP HTTP metrics endpoint from standard env vars.

    Resolution order (first non-empty value wins):

    1. ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`` (metrics-specific override)
    2. ``OTEL_EXPORTER_OTLP_ENDPOINT`` (general OTLP endpoint, SDK
       appends ``/v1/metrics``)
    3. ``""`` (empty -- OTLP metrics export disabled)

    Mirrors :func:`cosmos_curator.core.utils.infra.tracing.get_otlp_endpoint`
    so operators can configure traces and metrics with a single env
    var when desired.
    """
    return os.environ.get(_ENV_OTLP_METRICS_ENDPOINT) or os.environ.get(ENV_OTLP_ENDPOINT) or ""


@attrs.define(frozen=True)
class MetricsPushConfig:
    """Frozen configuration for the metrics push backend.

    Attributes:
        enabled: Master switch.  When ``False``,
            :func:`enable_metrics_push` is a no-op.
        otlp_endpoint: OTLP HTTP collector endpoint.  Empty string
            disables remote push (the scraper itself is not started).
            Mirrors the ``tracing_otlp_endpoint`` resolution shape.
        interval_seconds: Scrape + push interval.  Defaults to
            :data:`DEFAULT_INTERVAL_SECONDS` (30).
        scrape_timeout_seconds: Per-node HTTP GET timeout.
        service_name: OTel ``service.name`` resource attribute
            attached to all exported metrics.
        drop_regex: Optional regex applied to each OTel metric's
            name.  For compatibility with legacy counter patterns,
            matching also checks the ``_total``-stripped variant.
            Matching metrics are dropped pre-export, mirroring the
            ``metric_relabel_configs`` drop list used by the helm
            chart's collector sidecar.  Default is
            :data:`DEFAULT_DROP_METRIC_NAMES_REGEX`; pass ``""`` to
            disable filtering entirely.

    Note:
        The Ray Prometheus port is *not* configured here.  The scraper
        reads ``MetricsExportPort`` from each node's ``ray.nodes()``
        entry directly; nodes reporting ``0`` or missing are skipped
        (with a once-per-node warning).  In Slurm / NVCF / helm modes
        the port is already pinned by the bootstrap script via
        ``XENNA_RAY_METRICS_PORT`` -> ``ray start --metrics-export-port=``;
        in local mode Ray picks an ephemeral free port.  Either way,
        the scraper finds the right port without curator-side
        pinning, so we never force a port collision risk on the
        host.

    """

    enabled: bool = False
    otlp_endpoint: str = ""
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    scrape_timeout_seconds: int = DEFAULT_SCRAPE_TIMEOUT_SECONDS
    service_name: str = "cosmos_curator"
    drop_regex: str = DEFAULT_DROP_METRIC_NAMES_REGEX
    include_run_attributes: bool = True

    def compiled_drop_pattern(self) -> re.Pattern[str] | None:
        """Compile :attr:`drop_regex` once.  ``None`` when filtering disabled."""
        if not self.drop_regex:
            return None
        return re.compile(self.drop_regex)

    def resolved(self, *, cli_otlp_endpoint: str = "") -> "MetricsPushConfig":
        """Return a copy with OTLP endpoint and service name filled in.

        Endpoint precedence (mirrors :func:`enable_tracing` -- the
        CLI flag wins so an operator passing
        ``--otlp-metrics-push-endpoint`` is never silently overridden
        by a stale env var):

        1. ``cli_otlp_endpoint`` (CLI flag).
        2. :attr:`otlp_endpoint` on this config (if pre-set).
        3. ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT``.
        4. ``OTEL_EXPORTER_OTLP_ENDPOINT``.

        An empty result disables remote push.

        ``service.name`` is taken from ``OTEL_SERVICE_NAME`` when set,
        matching the tracing backend so traces and metrics share a
        single logical service in the collector.
        """
        endpoint = cli_otlp_endpoint.strip() or self.otlp_endpoint or get_otlp_metrics_endpoint()
        return attrs.evolve(
            self,
            otlp_endpoint=endpoint,
            service_name=os.environ.get(_ENV_SERVICE_NAME, self.service_name),
        )


def _is_connection_error(exc: BaseException) -> bool:
    """Walk the exception chain to detect connection-related failures.

    Matches the helper in :mod:`tracing_hook` so the resilient
    metrics exporter follows the same semantics: anything that smells
    like a connection refused / timeout / DNS failure is treated as
    a transient operator-side problem and used to disable push.
    """
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ConnectionError):
            return True
        if isinstance(current, OSError) and current.errno == errno.ECONNREFUSED:
            return True
        if isinstance(current, requests.exceptions.RequestException):
            return True
        current = current.__cause__ or current.__context__
    return False


class _ResilientMetricsExporter:
    """Wrap an OTLP metric exporter to suppress connection floods.

    Same idea as :class:`_ResilientOtlpExporter` in
    :mod:`tracing_hook`: after N consecutive connection-style
    failures (default :data:`_MAX_CONSECUTIVE_PUSH_FAILURES`) the
    exporter goes silent for the rest of the process lifetime,
    logging a single warning.  Non-connection errors are re-raised
    so the scrape loop can surface real bugs.

    Unlike traces, we do not have a local file fallback for metrics;
    when the exporter is disabled, the corresponding tick is simply
    dropped.  The pipeline keeps running.
    """

    def __init__(self, delegate: object) -> None:
        """Wrap *delegate* (an :class:`OTLPMetricExporter`) with failure suppression."""
        self._delegate = delegate
        self._disabled = False
        self._consecutive_failures = 0
        self._disabled_notice_logged = False
        # Guard ``_disabled`` and ``_consecutive_failures`` against
        # concurrent updates from the scrape loop thread and the
        # driver thread calling ``flush()``.  Worst case without
        # the lock is an off-by-one trip of the disable threshold,
        # but the lock is cheap and the export path is serialised
        # by the exporter's own network IO anyway.
        self._state_lock = threading.Lock()

    @property
    def disabled(self) -> bool:
        """Whether the exporter has been permanently disabled this run."""
        with self._state_lock:
            return self._disabled

    def export(self, metrics_data: MetricsData) -> bool:
        """Export *metrics_data* via the underlying OTLP exporter.

        Returns ``True`` on success or when silently disabled,
        ``False`` on a non-success exporter response that has not
        yet tripped the failure threshold.
        """
        with self._state_lock:
            if self._disabled:
                if not self._disabled_notice_logged:
                    self._disabled_notice_logged = True
                    logger.warning(
                        "[otel-metrics] OTLP metrics push is disabled for this process; "
                        "skipping further export attempts.",
                    )
                return True

        from opentelemetry.sdk.metrics.export import MetricExportResult  # noqa: PLC0415

        try:
            result = self._delegate.export(metrics_data)  # type: ignore[attr-defined]
        except Exception as exc:
            if _is_connection_error(exc):
                with self._state_lock:
                    self._consecutive_failures += 1
                    failures = self._consecutive_failures
                    if failures >= _MAX_CONSECUTIVE_PUSH_FAILURES:
                        self._disabled = True
                _log_otlp_push_failure(
                    failures=failures,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return False
            raise

        if result == MetricExportResult.SUCCESS:
            with self._state_lock:
                self._consecutive_failures = 0
            return True

        with self._state_lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            if failures >= _MAX_CONSECUTIVE_PUSH_FAILURES:
                self._disabled = True
        _log_otlp_push_failure(
            failures=failures,
            detail="exporter returned FAILURE (collector unreachable, rate limited, or auth rejected)",
        )
        return False

    def shutdown(self, timeout_millis: int = 30000) -> None:
        """Shut down the underlying exporter; suppress errors."""
        try:
            self._delegate.shutdown(timeout_millis=timeout_millis)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[otel-metrics] Exporter shutdown raised: {exc}")


def _parse_le(le: str) -> float:
    """Convert a Prometheus ``le`` label value to ``float``.

    ``+Inf`` is mapped to :data:`math.inf` so it sorts last; callers
    that build OTel ``explicit_bounds`` drop the ``+Inf`` entry
    (OTel infers the overflow bucket implicitly).
    """
    if le in ("+Inf", "Inf", "inf"):
        return math.inf
    try:
        return float(le)
    except ValueError:
        return math.inf


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Stable hashable key for a label-set, used by the start-time tracker."""
    return tuple(sorted(labels.items()))


def _point_labels(
    sample_labels: dict[str, str],
    series_labels: dict[str, str],
) -> dict[str, str]:
    """Merge run/node labels with scraped Prom sample labels.

    Sample labels win on key collision so Ray-provided tags such as
    ``node_type`` are never overwritten by empty run metadata.
    """
    if not series_labels:
        return dict(sample_labels)
    return {**series_labels, **sample_labels}


def _convert_gauge_like(
    family: object,
    *,
    now_unix_nano: int,
    series_labels: dict[str, str],
) -> list[Metric]:
    """Render gauge/unknown/info/stateset/gaugehistogram families as OTel gauges.

    Gauges are point-in-time observations, so ``start_time_unix_nano``
    is set to the current tick (per OTLP spec; only ``Sum`` and
    ``Histogram`` data points need a stable start time).
    """
    samples = family.samples  # type: ignore[attr-defined]
    data_points = [
        NumberDataPoint(
            attributes=_point_labels(dict(s.labels), series_labels),
            start_time_unix_nano=now_unix_nano,
            time_unix_nano=now_unix_nano,
            value=float(s.value),
        )
        for s in samples
    ]
    if not data_points:
        return []
    return [
        Metric(
            name=family.name,  # type: ignore[attr-defined]
            description=family.documentation,  # type: ignore[attr-defined]
            unit=family.unit,  # type: ignore[attr-defined]
            data=Gauge(data_points=data_points),
        ),
    ]


def _convert_summary(
    family: object,
    *,
    now_unix_nano: int,
    series_labels: dict[str, str],
) -> list[Metric]:
    """Render Prom ``summary`` samples as sample-name-specific gauges.

    Prom summaries are exposed as a mix of:

    * quantile samples (sample name == family name, with ``quantile`` label)
    * ``*_sum`` sample
    * ``*_count`` sample

    Mapping all of them to ``family.name`` collapses distinct series.
    Keep each sample's own name and group points by name.
    """
    samples = family.samples  # type: ignore[attr-defined]
    points_by_name: dict[str, list[NumberDataPoint]] = {}
    for sample in samples:
        metric_name = sample.name
        points_by_name.setdefault(metric_name, []).append(
            NumberDataPoint(
                attributes=_point_labels(dict(sample.labels), series_labels),
                start_time_unix_nano=now_unix_nano,
                time_unix_nano=now_unix_nano,
                value=float(sample.value),
            ),
        )
    return [
        Metric(
            name=metric_name,
            description=family.documentation,  # type: ignore[attr-defined]
            unit=family.unit,  # type: ignore[attr-defined]
            data=Gauge(data_points=data_points),
        )
        for metric_name, data_points in points_by_name.items()
        if data_points
    ]


def _convert_counter(
    family: object,
    *,
    now_unix_nano: int,
    start_time_for: SeriesStartTimeFn,
    series_labels: dict[str, str],
) -> list[Metric]:
    """Render a counter family as a cumulative monotonic OTel ``Sum``.

    Prom text exposes counters with a ``_total`` suffix on each sample
    name; we preserve this raw Prom name so dashboards and recording
    rules can be shared between scrape-based and OTLP-pushed paths.

    ``start_time_for(metric_name, label_key)`` returns the stable
    first-observation timestamp for the given series.  Using a stable
    start time (rather than the current tick) prevents OTLP backends
    from interpreting each push as a counter reset to zero.
    """
    samples = family.samples  # type: ignore[attr-defined]
    metric_name = next(
        (s.name for s in samples if s.name.endswith("_total")),
        family.name,  # type: ignore[attr-defined]
    )
    data_points: list[NumberDataPoint] = []
    for s in samples:
        if not s.name.endswith("_total"):
            continue
        labels = _point_labels(dict(s.labels), series_labels)
        start_t = start_time_for(metric_name, _label_key(labels))
        data_points.append(
            NumberDataPoint(
                attributes=labels,
                start_time_unix_nano=start_t,
                time_unix_nano=now_unix_nano,
                value=float(s.value),
            ),
        )
    if not data_points:
        return []
    return [
        Metric(
            name=metric_name,
            description=family.documentation,  # type: ignore[attr-defined]
            unit=family.unit,  # type: ignore[attr-defined]
            data=Sum(
                data_points=data_points,
                aggregation_temporality=AggregationTemporality.CUMULATIVE,
                is_monotonic=True,
            ),
        ),
    ]


@attrs.define
class _HistogramSlot:
    """Intermediate accumulator for one Prom histogram time series.

    Carries the per-``le`` cumulative bucket pairs and the matching
    ``_count`` / ``_sum`` aggregates while the parser walks every
    sample in a family.
    """

    attributes: dict[str, str]
    buckets: list[tuple[str, float]] = attrs.Factory(list)
    count: float | None = None
    sum: float | None = None


def _group_histogram_samples(
    family: object,
) -> dict[tuple[tuple[str, str], ...], _HistogramSlot]:
    """Bucket Prom histogram samples by their non-``le`` label set."""
    series: dict[tuple[tuple[str, str], ...], _HistogramSlot] = {}
    for sample in family.samples:  # type: ignore[attr-defined]
        base_labels = tuple(sorted((k, v) for k, v in sample.labels.items() if k != "le"))
        slot = series.setdefault(base_labels, _HistogramSlot(attributes=dict(base_labels)))
        if sample.name.endswith("_bucket"):
            le = sample.labels.get("le", "+Inf")
            slot.buckets.append((le, float(sample.value)))
        elif sample.name.endswith("_count"):
            slot.count = float(sample.value)
        elif sample.name.endswith("_sum"):
            slot.sum = float(sample.value)
    return series


def _histogram_data_point(
    slot: _HistogramSlot,
    *,
    now_unix_nano: int,
    start_time_unix_nano: int,
    series_labels: dict[str, str],
) -> HistogramDataPoint | None:
    """Build one OTel ``HistogramDataPoint`` from a grouped slot."""
    bucket_pairs = sorted(slot.buckets, key=lambda item: _parse_le(item[0]))
    if not bucket_pairs:
        return None
    explicit_bounds: list[float] = []
    cumulative_counts: list[int] = []
    for le, cum_value in bucket_pairs:
        cumulative_counts.append(int(cum_value))
        if le != "+Inf":
            explicit_bounds.append(_parse_le(le))
    per_bucket: list[int] = []
    prev = 0
    for cum in cumulative_counts:
        per_bucket.append(max(cum - prev, 0))
        prev = cum
    if len(per_bucket) == len(explicit_bounds):
        per_bucket.append(0)
    count_val = int(slot.count) if slot.count is not None else sum(per_bucket)
    sum_val = slot.sum if slot.sum is not None else 0.0
    return HistogramDataPoint(
        attributes=_point_labels(dict(slot.attributes), series_labels),
        start_time_unix_nano=start_time_unix_nano,
        time_unix_nano=now_unix_nano,
        count=count_val,
        sum=sum_val,
        bucket_counts=per_bucket,
        explicit_bounds=explicit_bounds,
        # Prometheus text format doesn't expose true min/max per
        # bucket -- pass None rather than fake zeros so the backend
        # doesn't render 0.0 as a real observation.  Verified that
        # the OTLP HTTP encoder accepts None and the wire format
        # simply omits these optional fields.
        min=None,  # type: ignore[arg-type]
        max=None,  # type: ignore[arg-type]
    )


def _convert_histogram(
    family: object,
    *,
    now_unix_nano: int,
    start_time_for: SeriesStartTimeFn,
    series_labels: dict[str, str],
) -> list[Metric]:
    """Convert a Prom ``histogram`` family into an OTel ``Histogram`` metric.

    Prometheus exposes histograms as a set of cumulative
    ``_bucket{le=...}`` samples plus ``_count`` and ``_sum``
    aggregates that share the family's base labels.  Each unique
    label set (minus ``le``) defines one OTel data point.

    Like counters, histogram buckets and sum/count are cumulative,
    so each data point uses a stable per-series ``start_time_unix_nano``
    via ``start_time_for(metric_name, label_key)``.
    """
    metric_name: str = family.name  # type: ignore[attr-defined]
    series = _group_histogram_samples(family)
    data_points: list[HistogramDataPoint] = []
    for label_key, slot in series.items():
        start_t = start_time_for(metric_name, label_key)
        point = _histogram_data_point(
            slot,
            now_unix_nano=now_unix_nano,
            start_time_unix_nano=start_t,
            series_labels=series_labels,
        )
        if point is not None:
            data_points.append(point)

    if not data_points:
        return []
    return [
        Metric(
            name=family.name,  # type: ignore[attr-defined]
            description=family.documentation,  # type: ignore[attr-defined]
            unit=family.unit,  # type: ignore[attr-defined]
            data=Histogram(
                data_points=data_points,
                aggregation_temporality=AggregationTemporality.CUMULATIVE,
            ),
        ),
    ]


def _convert_family(
    family: object,
    *,
    now_unix_nano: int,
    start_time_for: SeriesStartTimeFn,
    series_labels: dict[str, str],
) -> list[Metric]:
    """Dispatch one ``prometheus_client.Metric`` family to its OTel converter.

    Type mapping (Prometheus -> OTLP):

    * ``gauge`` -> :class:`Gauge` (single ``NumberDataPoint``).
    * ``counter`` -> :class:`Sum` with ``is_monotonic=True`` and
      cumulative temporality.
    * ``histogram`` -> :class:`Histogram` with cumulative buckets
      converted to per-bucket counts.  ``+Inf`` is dropped from
      explicit bounds (OTLP infers the implicit overflow bucket).
    * ``summary`` -> emitted as gauges using each sample's own name
      (quantiles on ``family.name`` with ``quantile`` label, plus
      ``*_sum`` / ``*_count`` as separate metrics).  OTel has no
      native Summary type.
    * ``unknown``/``info``/``stateset``/``gaugehistogram`` -> rendered
      as gauges, which keeps the data in the backend without
      pretending to a stronger type contract.

    ``start_time_for`` resolves the per-series first-observation
    timestamp for cumulative types (counter, histogram).
    """
    typ: str = family.type  # type: ignore[attr-defined]
    if typ == "counter":
        return _convert_counter(
            family,
            now_unix_nano=now_unix_nano,
            start_time_for=start_time_for,
            series_labels=series_labels,
        )
    if typ == "histogram":
        return _convert_histogram(
            family,
            now_unix_nano=now_unix_nano,
            start_time_for=start_time_for,
            series_labels=series_labels,
        )
    if typ == "summary":
        return _convert_summary(
            family,
            now_unix_nano=now_unix_nano,
            series_labels=series_labels,
        )
    # Gauge, summary, and anything else go through the same flat
    # gauge rendering path.
    return _convert_gauge_like(family, now_unix_nano=now_unix_nano, series_labels=series_labels)


def _matches_drop_pattern(metric_name: str, drop_pattern: re.Pattern[str]) -> bool:
    """Return True when ``drop_pattern`` matches metric name.

    For compatibility with older configs that targeted canonical OTel counter names,
    also check the name with a trailing ``_total`` removed.
    """
    if drop_pattern.fullmatch(metric_name):
        return True
    return metric_name.endswith("_total") and bool(drop_pattern.fullmatch(metric_name.removesuffix("_total")))


def _prom_text_to_metrics_data(  # noqa: PLR0913
    prom_text: str,
    *,
    resource_attrs: dict[str, str | int | float],
    series_labels: dict[str, str],
    scope_name: str,
    now_unix_nano: int,
    start_time_for: SeriesStartTimeFn,
    drop_pattern: re.Pattern[str] | None = None,
) -> MetricsData | None:
    """Convert Prometheus text exposition into an OTLP :class:`MetricsData`.

    Iterates over metric families produced by
    :func:`prometheus_client.parser.text_string_to_metric_families`
    and emits one OTel :class:`Metric` per family.  Returns ``None``
    when the input yields no metrics so the caller can skip the
    push entirely.

    Args:
        prom_text: Prometheus text exposition body.
        resource_attrs: OTel ``Resource`` attributes (typically
            ``service.name`` only; Mimir maps this to the Prom ``job``
            label).
        series_labels: Labels copied onto every exported data point so
            they appear as first-class Prom labels in Mimir/Grafana
            (e.g. ``slurm_job_id``, ``ray.node_id``).
        scope_name: ``InstrumentationScope.name`` for all metrics.
        now_unix_nano: Current tick timestamp for ``time_unix_nano``.
        start_time_for: Per-series first-observation timestamp
            resolver, keyed by ``(metric_name, sorted_label_pairs)``.
            Returned timestamp is used as ``start_time_unix_nano``
            for cumulative types (Sum, Histogram); gauges always
            use ``now_unix_nano``.
        drop_pattern: Optional pre-compiled ``re.Pattern`` matched
            against metric names. For counters, compatibility matching
            also checks the ``_total``-stripped variant so existing
            patterns keep working. Matching metrics are dropped
            pre-export to reduce backend cardinality. Pass ``None`` to
            keep all.

    Notes:
        * Bucket bounds in OTel must be sorted ascending and exclude
          the ``+Inf`` overflow bound.
        * Gauges use ``now_unix_nano`` for both timestamps (point-in-
          time semantics); only Sum / Histogram benefit from a stable
          start time to avoid counter-reset interpretation.

    """
    metrics: list[Metric] = []
    for family in text_string_to_metric_families(prom_text):
        family_metrics = _convert_family(
            family,
            now_unix_nano=now_unix_nano,
            start_time_for=start_time_for,
            series_labels=series_labels,
        )
        if drop_pattern is not None:
            family_metrics = [m for m in family_metrics if not _matches_drop_pattern(m.name, drop_pattern)]
        metrics.extend(family_metrics)

    if not metrics:
        return None

    resource = Resource.create(resource_attrs)
    scope = InstrumentationScope(name=scope_name, version="")
    scope_metrics = ScopeMetrics(scope=scope, metrics=metrics, schema_url="")
    resource_metrics = ResourceMetrics(
        resource=resource,
        scope_metrics=[scope_metrics],
        schema_url="",
    )
    return MetricsData(resource_metrics=[resource_metrics])


class MetricsPushBackend:
    """Per-driver metrics scraper + OTLP HTTP pusher.

    Lifecycle:

    1. :meth:`start` constructs the OTLP exporter (wrapped in
       :class:`_ResilientMetricsExporter`), arms a
       :class:`threading.Event` stop signal, and spawns the daemon
       loop thread.
    2. The loop runs :meth:`_scrape_and_push_once` every
       ``interval_seconds`` until the stop event fires or the thread
       is asked to exit.  Per-tick errors are caught and logged at
       DEBUG; the loop keeps going.
    3. :meth:`stop` sets the event, joins the thread (bounded), and
       shuts down the exporter.

    The thread is daemon=True so an unclean driver exit cannot leave
    the process hanging on a worker that lost its OTLP collector.

    Attributes:
        _config: Frozen :class:`MetricsPushConfig`.
        _exporter: Wrapped :class:`OTLPMetricExporter` or ``None``
            until :meth:`start` is called.
        _stop_event: Signalling event read by the loop.
        _thread: The daemon worker thread or ``None``.

    """

    def __init__(self, config: MetricsPushConfig) -> None:
        """Store the frozen *config* and initialize lifecycle fields."""
        self._config = config
        self._scrape_pid = str(os.getpid())
        self._scrape_tid = str(threading.get_ident())
        self._scrape_native_tid = str(threading.get_native_id())
        self._exporter: _ResilientMetricsExporter | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0
        self._last_tick_targets = 0
        # Node IDs that have already been logged as missing a
        # MetricsExportPort.  Used to suppress per-tick warning
        # spam when Ray reports 0 for a node on every scrape.
        self._missing_port_logged: set[str] = set()
        # Per-series first-observation timestamp.  Keyed by
        # (node_id, metric_name, sorted_label_pairs) so each Ray
        # node's counter is treated as its own time-series in the
        # backend.  Used by counter and histogram converters as
        # start_time_unix_nano so OTLP backends do not interpret
        # every push as a counter reset.
        self._series_start_times: dict[tuple[str, str, tuple[tuple[str, str], ...]], int] = {}
        # Protects _series_start_times, _missing_port_logged, and other
        # shared tick metadata.  Network scrape/export run outside it.
        self._tick_lock = threading.Lock()
        # Pre-compile the optional drop regex once.
        self._drop_pattern = config.compiled_drop_pattern()
        # Log once when the scrape loop stops because OTLP push was disabled.
        self._push_disabled_logged = False

    @property
    def tick_count(self) -> int:
        """Number of scrape ticks executed since :meth:`start`."""
        return self._tick_count

    def start(self) -> bool:
        """Build the exporter and spawn the scrape thread.

        No-op when ``config.enabled`` is ``False`` or ``otlp_endpoint``
        is empty (the latter mirrors the trace exporter behaviour --
        no endpoint, no remote push).
        """
        if not self._config.enabled or not self._config.otlp_endpoint:
            logger.debug(
                "[otel-metrics] start() skipped: "
                f"enabled={self._config.enabled} endpoint={self._config.otlp_endpoint!r}",
            )
            return False

        # Resolve endpoint explicitly for OTLPMetricExporter so a stale
        # env var cannot silently override the CLI/config endpoint.
        # Metrics-specific endpoint is used verbatim.  Generic endpoint
        # receives "/v1/metrics" exactly once.
        metrics_env = os.environ.get(_ENV_OTLP_METRICS_ENDPOINT, "").strip()
        generic_env = os.environ.get(ENV_OTLP_ENDPOINT, "").strip()
        exporter_endpoint = metrics_env or _ensure_metrics_path(generic_env or self._config.otlp_endpoint)

        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: PLC0415
            OTLPMetricExporter,
        )

        try:
            delegate = OTLPMetricExporter(endpoint=exporter_endpoint)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[otel-metrics] Failed to construct OTLPMetricExporter: {exc}", exc_info=True)
            return False

        self._exporter = _ResilientMetricsExporter(delegate)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="cosmos-curator-metrics-push",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"[otel-metrics] Started scraper thread "
            f"(interval={self._config.interval_seconds}s, "
            f"endpoint={self._config.otlp_endpoint})",
        )
        return True

    def stop(self, join_timeout: float = 5.0) -> None:
        """Signal the loop, join the thread, and shut down the exporter.

        Bounded join keeps shutdown predictable even when a final
        push is still in flight.  Idempotent: calling twice is safe.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logger.debug("[otel-metrics] Scrape thread did not exit within join timeout")
        self._thread = None
        if self._exporter is not None:
            self._exporter.shutdown()
            self._exporter = None
        logger.debug(f"[otel-metrics] Stopped after {self._tick_count} tick(s)")

    def flush(self) -> None:
        """Execute one immediate scrape + push tick.

        Used by :func:`flush_metrics_push` to capture the final
        cumulative counter values just before Ray shuts down.  Safe
        to call even when the loop thread is still running -- the
        OTel exporter is thread-safe and an extra data point is
        always preferable to a lost one at shutdown time.
        """
        if self._exporter is None or self._exporter.disabled:
            return
        try:
            self._scrape_and_push_once()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[otel-metrics] flush tick failed: {exc}")

    def _run_loop(self) -> None:
        """Daemon thread entry point.  Catches everything to stay alive."""
        try:
            while not self._stop_event.is_set():
                try:
                    self._scrape_and_push_once()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"[otel-metrics] Scrape tick raised: {exc}")
                # event.wait returns True when the event fires, so we
                # exit promptly on stop().
                if self._stop_event.wait(self._config.interval_seconds):
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[otel-metrics] Scrape loop exited unexpectedly: {exc}", exc_info=True)

    def _log_push_disabled_once(self) -> None:
        """Emit the one-time warning when remote push has been disabled."""
        with self._tick_lock:
            if self._push_disabled_logged:
                return
            self._push_disabled_logged = True
        logger.warning(
            "[otel-metrics] Remote OTLP metrics push is disabled; "
            "scrape/push ticks will be skipped for the rest of this run.",
        )

    def _scrape_and_push_once(self) -> None:
        """Enumerate live Ray nodes, scrape each, then push the merged result."""
        with self._tick_lock:
            self._tick_count += 1

        exporter = self._exporter
        if exporter is None:
            return
        if exporter.disabled:
            self._log_push_disabled_once()
            return

        targets = self._enumerate_targets()
        with self._tick_lock:
            self._last_tick_targets = len(targets)
        if not targets:
            logger.debug("[otel-metrics] No live Ray nodes to scrape this tick")
            return

        now_unix_nano = time.time_ns()
        for node_id, url, node_host in targets:
            prom_text = self._scrape_node(url)
            if not prom_text:
                continue
            instance_label = self._instance_label_for(url, node_host=node_host, node_id=node_id)
            start_time_for = self._make_start_time_for(node_id, now_unix_nano)
            try:
                metrics_data = _prom_text_to_metrics_data(
                    prom_text,
                    resource_attrs=self._resource_attrs_for(service_instance_id=instance_label),
                    series_labels=self._series_labels_for(node_id, url, node_host),
                    scope_name="cosmos_curator.ray_metrics",
                    now_unix_nano=now_unix_nano,
                    start_time_for=start_time_for,
                    drop_pattern=self._drop_pattern,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[otel-metrics] Failed to convert metrics from {url}: {exc}")
                continue
            if metrics_data is None:
                continue
            if exporter.export(metrics_data):
                logger.info(
                    f"[otel-metrics] push ok ts_ns={time.time_ns()} tick={self._tick_count} "
                    f"targets={len(targets)} target={url}",
                )

    def _make_start_time_for(
        self,
        node_id: str,
        now_unix_nano: int,
    ) -> SeriesStartTimeFn:
        """Return a closure that resolves first-seen times for this node.

        The closure looks up
        ``(node_id, metric_name, label_key)`` in
        :attr:`_series_start_times`; on miss it records ``now_unix_nano``
        and returns it.  The series cache is guarded by
        :attr:`_tick_lock` inside the closure.
        """

        def _resolve(metric_name: str, label_key: tuple[tuple[str, str], ...]) -> int:
            key = (node_id, metric_name, label_key)
            with self._tick_lock:
                existing = self._series_start_times.get(key)
                if existing is not None:
                    return existing
                self._series_start_times[key] = now_unix_nano
                return now_unix_nano

        return _resolve

    def _resource_attrs_for(self, *, service_instance_id: str = "") -> dict[str, str | int | float]:
        """Build minimal OTel ``Resource`` attributes for each export batch."""
        attrs: dict[str, str | int | float] = {"service.name": self._config.service_name}
        if service_instance_id:
            attrs["service.instance.id"] = service_instance_id
        return attrs

    def _series_labels_for(
        self,
        node_id: str,
        url: str,
        node_host: str,
    ) -> dict[str, str]:
        """Build Prom-style labels attached to every scraped sample.

        Run and node metadata live here (not on the OTel resource) so
        Mimir exposes them as queryable labels on the metric series.
        """
        labels: dict[str, str] = {
            "host.name": short_host_label(node_host),
            "ray.node_id": node_id,
            "ray.metrics_url": url,
            "scrape_pid": self._scrape_pid,
            "scrape_tid": self._scrape_tid,
            "scrape_native_tid": self._scrape_native_tid,
        }
        if self._config.include_run_attributes:
            labels.update(collect_run_attributes())
        return labels

    @staticmethod
    def _instance_label_for(url: str, *, node_host: str, node_id: str) -> str:
        """Best-effort stable ``instance`` label that never affects scrape flow."""
        parsed = urlsplit(url)
        if parsed.netloc:
            return parsed.netloc
        return node_host or node_id or "unknown"

    def _enumerate_targets(self) -> list[tuple[str, str, str]]:
        """Return ``(node_id, metrics_url, node_host)`` for each live Ray node.

        Reads ``MetricsExportPort`` directly from each
        ``ray.nodes()`` entry, so it works in both local mode (Ray
        picks an ephemeral port) and Slurm / NVCF / helm modes
        (bootstrap script pinned the port).  Nodes reporting ``0``
        or a missing port indicate Ray's metrics exporter never
        came up on that node -- they are logged once at WARN per
        node ID and skipped, then logged at DEBUG on subsequent
        ticks to avoid flooding.

        Errors querying ``ray.nodes()`` (e.g. Ray not yet up, GCS
        glitch) return an empty list rather than raising; the next
        tick will retry.
        """
        try:
            import ray  # noqa: PLC0415

            nodes = ray.nodes()  # type: ignore[no-untyped-call]
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[otel-metrics] ray.nodes() raised: {exc}")
            return []

        targets: list[tuple[str, str, str]] = []
        for node in nodes:
            if not node.get("Alive"):
                continue
            scrape_host = node.get("NodeManagerAddress") or node.get("NodeManagerHostname")
            if not scrape_host:
                continue
            # Prefer hostnames for labels when available; in Kubernetes
            # NodeManagerAddress is often a pod IP and less readable.
            label_host = node.get("NodeManagerHostname") or scrape_host
            port = node.get("MetricsExportPort")
            if not port:
                # Log at WARN once per node, then DEBUG thereafter --
                # a node whose metrics exporter never started will
                # report 0 on every tick, which would otherwise flood
                # the logs.
                node_id = node.get("NodeID", label_host)
                with self._tick_lock:
                    first_missing = node_id not in self._missing_port_logged
                    if first_missing:
                        self._missing_port_logged.add(node_id)
                if first_missing:
                    logger.warning(
                        f"[otel-metrics] Skipping node {node_id}: MetricsExportPort missing/0 "
                        f"(Ray's Prometheus exporter did not start on this node). "
                        f"Subsequent occurrences logged at DEBUG.",
                    )
                else:
                    logger.debug(f"[otel-metrics] Still skipping node {node_id}: MetricsExportPort missing/0.")
                continue
            url = f"http://{scrape_host}:{port}/metrics"
            targets.append((node.get("NodeID", label_host), url, label_host))
        return targets

    def _scrape_node(self, url: str) -> str:
        """HTTP GET *url* and return the body, or empty string on failure."""
        try:
            response = requests.get(url, timeout=self._config.scrape_timeout_seconds)
        except requests.RequestException as exc:
            logger.debug(f"[otel-metrics] Scrape failed for {url}: {exc}")
            return ""
        if not response.ok:
            logger.debug(f"[otel-metrics] Scrape returned HTTP {response.status_code} for {url}")
            return ""
        return response.text


# Per-process singleton.  Set by enable_metrics_push(), read by
# disable_metrics_push() / flush_metrics_push().  None when metrics
# push is not configured.
_backend_lock = threading.Lock()
_current_backend: MetricsPushBackend | None = None


def enable_metrics_push(
    config: MetricsPushConfig,
    *,
    cli_otlp_endpoint: str = "",
) -> bool:
    """Start the driver-side metrics scraper.

    Args:
        config: Push settings (call :meth:`MetricsPushConfig.resolved`
            first, or pass ``cli_otlp_endpoint`` here).
        cli_otlp_endpoint: OTLP URL from the CLI flag.  When non-empty
            it is written to ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT``.
            We intentionally do not overwrite the generic
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` here because tracing may
            have already pinned a different endpoint.

    Returns:
        True only when this call started a backend; False when it was a
        no-op (disabled, no endpoint, a backend already active, or start
        failed).

    Note:
        No-op when ``config.enabled`` is ``False`` or the resolved OTLP
        endpoint is empty.

        The Ray Prometheus metrics port is **not** touched here.  In
        local mode Ray binds its own ephemeral free port; in
        Slurm / NVCF / helm modes the bootstrap script chose the port at
        ``ray start --metrics-export-port=`` time (via
        ``XENNA_RAY_METRICS_PORT``).  The scraper reads each node's
        actual port from ``ray.nodes()[i].MetricsExportPort`` at scrape
        time, so we deliberately avoid forcing a specific port via
        ``ray.init(_metrics_export_port=...)`` -- doing so would only add
        a port-collision risk in local mode without buying us anything.

    """
    global _current_backend  # noqa: PLW0603

    set_otlp_run_attributes_enabled(enabled=config.include_run_attributes)

    cli_endpoint = cli_otlp_endpoint.strip()
    if cli_endpoint:
        # Make the CLI-supplied metrics endpoint authoritative without
        # mutating OTEL_EXPORTER_OTLP_ENDPOINT, which tracing may use.
        os.environ[_ENV_OTLP_METRICS_ENDPOINT] = _ensure_metrics_path(cli_endpoint)

    resolved = config.resolved(cli_otlp_endpoint=cli_otlp_endpoint)

    if not resolved.enabled:
        return False

    if not resolved.otlp_endpoint:
        logger.info(
            "[otel-metrics] enable_metrics_push requested but no OTLP "
            "endpoint resolved (set --otlp-metrics-push-endpoint or "
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT/OTEL_EXPORTER_OTLP_ENDPOINT)",
        )
        return False

    # Re-entrancy guard.  enable_metrics_push() must be idempotent
    # because profiling_scope() and some test fixtures may call it
    # more than once on the driver.
    with _backend_lock:
        if _current_backend is not None:
            return False
        backend = MetricsPushBackend(resolved)
        if not backend.start():
            return False
        _current_backend = backend
        return True


def disable_metrics_push(join_timeout: float = 5.0) -> None:
    """Stop the scraper and release the module singleton.

    Registered as a pre-shutdown hook by ``profiling_scope`` so it
    runs before ``ray.shutdown()`` (LIFO order in
    :func:`shutdown_cluster`).  Idempotent.
    """
    global _current_backend  # noqa: PLW0603
    with _backend_lock:
        if _current_backend is None:
            return
        backend = _current_backend
        _current_backend = None
    backend.stop(join_timeout=join_timeout)


def flush_metrics_push() -> None:
    """Force one synchronous scrape + push tick.

    Useful to capture the very last counter values before Ray
    shuts down.  No-op when push is not currently active.
    """
    with _backend_lock:
        backend = _current_backend
    if backend is None:
        return
    backend.flush()


@contextmanager
def otlp_push_scope() -> Iterator[None]:
    """Run a pipeline under the env-driven driver-side OTLP metrics push.

    Interval and drop list come from ``COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL``
    and ``COSMOS_CURATOR_OTLP_METRICS_PUSH_DROP_REGEX``; the endpoint is
    resolved from the standard ``OTEL_*`` env vars. A misconfiguration warns
    and runs without the push; it never fails the run. Only a backend this
    scope started is stopped, and the final flush precedes the stop so
    terminal counter values are exported. Run attributes honor
    ``COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES``; the scope forwards it, never overrides.
    """
    interval = env_int(OTLP_METRICS_PUSH_INTERVAL_ENV_VAR, minimum=1) or DEFAULT_INTERVAL_SECONDS
    drop_regex = os.environ.get(_ENV_DROP_REGEX, DEFAULT_DROP_METRIC_NAMES_REGEX)
    started = False
    try:
        started = enable_metrics_push(
            MetricsPushConfig(
                enabled=True,
                interval_seconds=interval,
                drop_regex=drop_regex,
                include_run_attributes=otlp_run_attributes_enabled(),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=True).warning(f"[otel-metrics] Failed to start metrics push: {exc}")
    try:
        yield
    finally:
        if started:
            try:
                flush_metrics_push()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[otel-metrics] Failed to flush metrics push: {exc}")
            try:
                disable_metrics_push()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[otel-metrics] Failed to disable metrics push: {exc}")


__all__: Sequence[str] = (
    "DEFAULT_DROP_METRIC_NAMES_REGEX",
    "DEFAULT_INTERVAL_SECONDS",
    "DEFAULT_SCRAPE_TIMEOUT_SECONDS",
    "MetricsPushBackend",
    "MetricsPushConfig",
    "disable_metrics_push",
    "enable_metrics_push",
    "flush_metrics_push",
    "get_otlp_metrics_endpoint",
    "otlp_push_scope",
)
