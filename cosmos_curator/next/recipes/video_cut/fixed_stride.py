# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-stride cutting: probe each source's duration and tile it into spans.

Each task first probes a presigned URL with FFprobe, using HTTP ranges where
possible. If that is inconclusive, the same task streams the source to a
temporary file and probes it locally before planning. The CPU reservation
bounds task concurrency; source bytes stay out of Ray and no download actor
pool is needed. Spans depend only on the duration.
"""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pyarrow as pa
import ray
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.next.media.failures import MediaError, ProbeError
from cosmos_curator.next.media.ffmpeg import VideoMetadata, probe_video_path, probe_video_source
from cosmos_curator.next.media.spans import fixed_stride_spans, seconds_to_nanoseconds
from cosmos_curator.next.recipes.video_cut.config import ExecutionConfig, FixedStrideConfig, ResolvedVideoCutConfig
from cosmos_curator.next.recipes.video_cut.contracts import (
    SOURCE_SCHEMA,
    error_result,
    media_fields,
    source_result,
)
from cosmos_curator.next.recipes.video_cut.sources import unreadable_source_error
from cosmos_curator.next.utils.source_objects import s3_client


def probe_source(uri: str, *, execution: ExecutionConfig) -> tuple[VideoMetadata, int]:
    """Probe remotely first, confirming inconclusive results against a complete local download."""
    client = s3_client(execution.storage_profile)
    prefix = S3Prefix(uri)
    timeout = execution.probe_timeout_s
    size = int(client.s3.head_object(Bucket=prefix.bucket, Key=prefix.prefix)["ContentLength"])
    signed = client.s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": prefix.bucket, "Key": prefix.prefix},
        ExpiresIn=max(900, timeout * 2),
    )
    try:
        return probe_video_source(signed, timeout_s=timeout), size
    except (MediaError, ProbeError):
        # A remote failure cannot establish whether the source itself is bad.
        pass
    with TemporaryDirectory(prefix="curator_video_cut_") as directory:
        path = Path(directory) / "source.mp4"
        client.download_to_path(prefix, path)
        return probe_video_path(path, timeout_s=timeout), path.stat().st_size


def planned_source(source: Mapping[str, Any], split: FixedStrideConfig) -> dict[str, Any]:
    """Tile a probed source's duration into fixed-stride spans, including a successful empty plan."""
    spans = fixed_stride_spans(
        source["source_duration_ns"],
        duration_ns=seconds_to_nanoseconds(split.duration_s),
        stride_ns=seconds_to_nanoseconds(split.stride_s),
        min_duration_ns=seconds_to_nanoseconds(split.min_duration_s),
    )
    return source_result(source, spans)


def plan_sources(batch: pa.Table, *, split: FixedStrideConfig, execution: ExecutionConfig) -> pa.Table:
    """Probe and plan each source within one task; only terminal outcomes reach checkpoints."""
    results = []
    for source in batch.to_pylist():
        uri = source["source_uri"]
        try:
            metadata, size_bytes = probe_source(uri, execution=execution)
        except ClientError as exc:
            error = unreadable_source_error(exc, uri, storage_profile=execution.storage_profile)
            if error is None:
                raise
            results.append(error_result(source, **error))
        except MediaError as exc:
            # Inconclusive ProbeErrors propagate; only a complete local probe settles bad media.
            results.append(error_result(source, error_stage="source-probe", error_message=str(exc)))
        else:
            results.append(planned_source(source | media_fields(metadata, size_bytes), split))
    return pa.Table.from_pylist(results, schema=SOURCE_SCHEMA)


def plan(sources: ray.data.Dataset, config: ResolvedVideoCutConfig) -> ray.data.Dataset:
    """Probe and plan on task workers, with each source's local fallback inside the same task.

    The task sets no remote args besides ``num_cpus``, so a read reserving the
    same CPUs fuses into it. Inconclusive probes retry through the map retries
    that ``configure_ray_data_stability`` enables.
    """
    assert isinstance(config.split, FixedStrideConfig)
    return sources.map_batches(
        cast("Any", plan_sources),
        batch_size=None,
        batch_format="pyarrow",
        fn_kwargs={"split": config.split, "execution": config.execution},
        num_cpus=config.execution.fixed_stride.probe_cpus,
    )
