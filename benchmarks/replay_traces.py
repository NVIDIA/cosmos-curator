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

r"""Replay OTel trace ``.jsonl`` files to an OTLP endpoint.

Standalone CLI tool.  Run after a pipeline execution to re-export the
``.jsonl`` span files produced by ``--profile-tracing`` into Jaeger,
Grafana Tempo, VictoriaTraces, or any OTLP-compatible backend.

The script appends ``/v1/traces`` to the URL you pass.  Backends that use
a different path (e.g. VictoriaTraces) require that path in the base URL.

Quick start -- Jaeger all-in-one (simplest)
-------------------------------------------

::

    # 1. Start Jaeger (OTLP HTTP on 4318, UI on 16686)
    docker run -d --name jaeger \
      -p 16686:16686 \
      -p 4318:4318 \
      jaegertracing/jaeger:latest \
      --set receivers.otlp.protocols.http.endpoint=0.0.0.0:4318

    # 2. Replay traces
    python -m benchmarks.replay_traces tmp/traces/ \
        --replay-to http://localhost:4318


    # 3. In Grafana UI (http://localhost:3000):
    #    Connections > Data sources > Add Jaeger
    #    URL: http://host.docker.internal:16686

    # 4. Open http://localhost:16686, select service "cosmos_curator"

Quick start -- Grafana + Tempo (richer dashboards)
--------------------------------------------------

::

    # 1. Create minimal Tempo config
    cat > /tmp/tempo.yaml <<'EOF'
    server:
      http_listen_port: 3200
    distributor:
      receivers:
        otlp:
          protocols:
            http:
              endpoint: "0.0.0.0:4318"
    ingester:
      max_block_duration: 5s   # flush WAL quickly (default 30m)
    query_frontend:
      search:
        max_duration: 0         # no limit on search time range
    storage:
      trace:
        backend: local
        local:
          path: /var/tempo/traces
        wal:
          path: /var/tempo/wal
    EOF

    # 2. Start Tempo (OTLP HTTP on 4318, API on 3200)
    #    Pin to 2.6 -- v2.10+ requires Kafka distributor config.
    docker run -d --name tempo \
      -p 3200:3200 -p 4318:4318 \
      -v /tmp/tempo.yaml:/etc/tempo.yaml \
      grafana/tempo:2.6.1 \
      -config.file=/etc/tempo.yaml

    # 3. Start Grafana (UI on 3000, auth disabled)
    docker run -d --name grafana \
      -p 3000:3000 \
      -e GF_AUTH_ANONYMOUS_ENABLED=true \
      -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
      -e GF_AUTH_DISABLE_LOGIN_FORM=true \
      grafana/grafana:latest

    # 4. In Grafana UI (http://localhost:3000):
    #    Connections > Data sources > Add Tempo
    #    URL: http://host.docker.internal:3200

    # 5. Replay traces
    python -m benchmarks.replay_traces tmp/traces/ \
        --replay-to http://localhost:4318

Cleanup::

    docker rm -f jaeger tempo grafana

Usage examples
--------------

::

    # Replay to a local Jaeger (standard OTLP path)
    python -m benchmarks.replay_traces ./profiles/traces/ \
        --replay-to http://localhost:4318

    # Replay to VictoriaTraces (path is /insert/opentelemetry/v1/traces)
    python -m benchmarks.replay_traces ./profiles/traces/ \
        --replay-to http://localhost:10428/insert/opentelemetry

    # Filter by span name
    python -m benchmarks.replay_traces ./profiles/traces/ \
        --replay-to http://localhost:4318 --filter-name VideoDownloader

    # HTTP 400 request body too large: use a smaller export batch
    python -m benchmarks.replay_traces ./profiles/traces/ \
        --replay-to http://localhost:4318 --export-batch-size 128

"""

import argparse
import json
import pathlib
import sys
from datetime import datetime

from loguru import logger


def _discover_jsonl(directory: pathlib.Path) -> list[pathlib.Path]:
    """Glob ``.jsonl`` span files in *directory*, sorted by name."""
    if not directory.is_dir():
        msg = f"Directory does not exist: {directory}"
        raise FileNotFoundError(msg)
    return sorted(directory.glob("*.jsonl"))


def _parse_spans(path: pathlib.Path) -> list[dict]:
    """Parse all spans from a single ``.jsonl`` file."""
    spans: list[dict] = []
    with path.open() as f:
        for line_no, raw_line in enumerate(f, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                spans.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                logger.warning(f"  Skipping {path.name}:{line_no}: {e}")
    return spans


def _parse_timestamp(ts: str | None) -> datetime | None:
    """Parse an OTel ISO-8601 timestamp string."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _hex_to_int(hex_str: str | None) -> int:
    """Convert a ``0x...`` hex string to int, or 0 if falsy.

    OTel trace IDs are 128-bit and span IDs are 64-bit.  The OTLP
    protobuf exporter converts the int back to fixed-width bytes
    via ``int.to_bytes(16, 'big')``, so leading zeros are preserved
    in the wire format even though ``int()`` drops them.

    """
    if not hex_str:
        return 0
    return int(hex_str, 16)


def _iso_to_ns(ts: str | None) -> int | None:
    """Convert ISO-8601 timestamp to nanoseconds since epoch."""
    dt = _parse_timestamp(ts)
    if dt is None:
        return None
    return int(dt.timestamp() * 1_000_000_000)


def _json_to_readable_span(span_json: dict) -> object | None:
    """Convert a single span JSON dict to an OTel ``ReadableSpan``.

    Returns ``None`` if the span is missing required context IDs.

    """
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    from opentelemetry.sdk.trace import ReadableSpan  # noqa: PLC0415
    from opentelemetry.trace import SpanContext, SpanKind, StatusCode, TraceFlags  # noqa: PLC0415
    from opentelemetry.trace.status import Status  # noqa: PLC0415

    status_map = {
        "OK": StatusCode.OK,
        "ERROR": StatusCode.ERROR,
        "UNSET": StatusCode.UNSET,
    }

    kind_map: dict[str | int, SpanKind] = {
        "SpanKind.INTERNAL": SpanKind.INTERNAL,
        "SpanKind.SERVER": SpanKind.SERVER,
        "SpanKind.CLIENT": SpanKind.CLIENT,
        "SpanKind.PRODUCER": SpanKind.PRODUCER,
        "SpanKind.CONSUMER": SpanKind.CONSUMER,
        0: SpanKind.INTERNAL,
        1: SpanKind.SERVER,
        2: SpanKind.CLIENT,
        3: SpanKind.PRODUCER,
        4: SpanKind.CONSUMER,
    }

    ctx = span_json.get("context", {})
    trace_id = _hex_to_int(ctx.get("trace_id"))
    span_id = _hex_to_int(ctx.get("span_id"))

    if trace_id == 0 or span_id == 0:
        return None

    parent_id = _hex_to_int(span_json.get("parent_id"))
    # TraceFlags.SAMPLED (0x01) is required so that BatchSpanProcessor and
    # downstream backends treat the span as sampled rather than silently
    # dropping it. Without this flag the replay appears to succeed but no
    # spans show up in the collector / viewer.
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )
    parent_context = (
        SpanContext(trace_id=trace_id, span_id=parent_id, is_remote=False, trace_flags=TraceFlags.SAMPLED)
        if parent_id
        else None
    )

    # Status
    status_data = span_json.get("status", {})
    status_code_str = status_data.get("status_code", "UNSET") if isinstance(status_data, dict) else "UNSET"
    status_desc = status_data.get("description") if isinstance(status_data, dict) else None
    status = Status(status_map.get(status_code_str, StatusCode.UNSET), status_desc)

    # Resource
    resource_data = span_json.get("resource", {})
    resource_attrs = resource_data.get("attributes", {}) if isinstance(resource_data, dict) else {}

    return ReadableSpan(
        name=span_json.get("name", "<unnamed>"),
        context=span_context,
        parent=parent_context,
        resource=Resource(attributes=resource_attrs),
        attributes=span_json.get("attributes", {}) or {},
        kind=kind_map.get(span_json.get("kind"), SpanKind.INTERNAL),
        status=status,
        start_time=_iso_to_ns(span_json.get("start_time")),
        end_time=_iso_to_ns(span_json.get("end_time")),
    )


def _replay_to_otlp(all_spans: list[dict], endpoint: str, export_batch_size: int) -> bool:
    """Replay parsed spans to an OTLP HTTP endpoint.

    Converts raw span dicts into ``ReadableSpan`` objects and exports
    them synchronously using ``OTLPSpanExporter.export()`` in chunks of
    at most *export_batch_size* spans per HTTP request.  Many receivers
    reject a single giant protobuf payload (HTTP 400: request body too
    large); chunking keeps each POST under typical limits.

    ::

        .jsonl dicts --> _json_to_readable_span --> OTLPSpanExporter.export()
                                    (chunked)              |
                                                           v
                                                     OTLP endpoint

    A synchronous (direct) export is used instead of ``BatchSpanProcessor``
    because replay is a one-shot operation: we want immediate confirmation
    that all spans were delivered and clear error reporting on failure.

    Requires ``opentelemetry-exporter-otlp-proto-http``.

    Returns:
        False if any export batch failed.  True if there were no valid spans
        to send, or every batch exported successfully.

    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import SpanExportResult  # noqa: PLC0415

    readable_spans = []
    skipped = 0

    for span_json in all_spans:
        try:
            readable = _json_to_readable_span(span_json)
            if readable is None:
                skipped += 1
                continue
            readable_spans.append(readable)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"  Skipping span: {e}")
            skipped += 1

    if skipped:
        logger.info(f"Skipped {skipped} malformed/empty span(s)")

    if not readable_spans:
        logger.warning("No valid spans to replay")
        return True

    logger.info(f"Replaying {len(readable_spans)} span(s) to {endpoint} ({export_batch_size} span(s) per HTTP request)")

    # OTLPSpanExporter uses the endpoint parameter as-is (no auto-appending
    # of /v1/traces), so we must include the full path ourselves.
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")

    n = len(readable_spans)
    batch_idx = 0
    for start in range(0, n, export_batch_size):
        batch_idx += 1
        chunk = readable_spans[start : start + export_batch_size]
        result = exporter.export(chunk)
        if result != SpanExportResult.SUCCESS:
            end = start + len(chunk) - 1
            logger.error(f"Replay failed on batch {batch_idx} (span indices {start}-{end}): exporter returned {result}")
            exporter.shutdown()
            return False

    logger.info(f"Replay complete: {n} span(s) exported in {batch_idx} request(s)")
    exporter.shutdown()
    return True


def main(argv: list[str] | None = None) -> None:
    """Entry point for the trace replay CLI."""
    parser = argparse.ArgumentParser(
        description="Replay OTel trace .jsonl files to an OTLP endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", type=pathlib.Path, help="Directory containing .jsonl span files.")
    parser.add_argument(
        "--replay-to",
        type=str,
        required=True,
        metavar="URL",
        help=(
            "OTLP HTTP endpoint base URL. /v1/traces is appended. "
            "Jaeger/Tempo: http://localhost:4318 "
            "VictoriaTraces: http://localhost:10428/insert/opentelemetry"
        ),
    )
    parser.add_argument(
        "--filter-name",
        type=str,
        default=None,
        metavar="PATTERN",
        help="Only include spans whose name contains this substring.",
    )
    parser.add_argument(
        "--export-batch-size",
        type=int,
        default=512,
        metavar="N",
        help=(
            "Maximum spans per OTLP HTTP export request (payload size is not fixed; "
            "attribute-heavy spans need a smaller value). "
            "Lower this if the receiver returns HTTP 400 (request body too large)."
        ),
    )

    args = parser.parse_args(argv)

    # 1. Discover
    paths = _discover_jsonl(args.directory)
    if not paths:
        logger.error(f"No .jsonl files found in {args.directory}")
        sys.exit(1)
    logger.info(f"Found {len(paths)} span file(s)")

    # 2. Parse
    all_spans: list[dict] = []
    for p in paths:
        all_spans.extend(_parse_spans(p))

    if not all_spans:
        logger.error("No spans found")
        sys.exit(1)

    # 3. Filter
    if args.filter_name:
        before = len(all_spans)
        all_spans = [s for s in all_spans if args.filter_name in s.get("name", "")]
        logger.info(f"Filter '{args.filter_name}': {before} -> {len(all_spans)} span(s)")
        if not all_spans:
            logger.error(f"No spans match filter '{args.filter_name}'")
            sys.exit(1)

    logger.info(f"Total: {len(all_spans)} span(s)")

    if args.export_batch_size < 1:
        logger.error("--export-batch-size must be >= 1")
        sys.exit(1)

    # 4. Replay
    if not _replay_to_otlp(all_spans, args.replay_to, args.export_batch_size):
        sys.exit(1)


if __name__ == "__main__":
    main()
