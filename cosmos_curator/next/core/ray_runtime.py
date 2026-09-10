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

"""Shared Ray Data runtime configuration for Curator Next recipes."""

import os
from collections.abc import Mapping

import ray

from cosmos_curator.core.utils import environment

# Substring patterns matched against ``"ClassName: message"`` (via
# ``ray._common.retry.format_exception``) to decide whether a map function
# error is transient and worth retrying. Class-name substrings avoid
# dragging in botocore just to reference exception types.
#
# ``botocore.exceptions.ClientError`` is deliberately excluded because it
# is the base class for *all* AWS HTTP responses -- including deterministic
# 4xx errors (NoSuchKey, AccessDenied, InvalidBucketName) that should fail
# fast, not consume ``_MAP_MAX_RETRIES`` attempts. Botocore already retries
# transient 5xx / throttling at its own HTTP client layer; what still
# bubbles up to us is caught via the transport-level patterns below
# (DNS / URL, socket read stalls, mid-stream drops on large GETs).
_MAP_RETRY_PATTERNS: list[str] = [
    "OSError",
    "ConnectionError",
    "TimeoutError",
    "EndpointConnectionError",
    "ReadTimeoutError",
    "IncompleteRead",
]
_MAP_MAX_RETRIES = 3


def configure_ray_data_progress(*, progress: bool) -> None:
    """Configure Ray Data progress output before creating datasets."""
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = progress
    ctx.enable_operator_progress_bars = progress
    ctx.enable_rich_progress_bars = progress
    ctx.print_on_execution_start = progress
    ctx.use_ray_tqdm = False


def configure_ray_data_stability(
    *,
    disable_high_memory_detector: bool = False,
    no_progress_timeout_s: float | None = None,
) -> None:
    """Enable Ray 2.56 stability knobs for map-heavy Ray Data pipelines.

    - ``default_map_logical_memory_enabled`` (PR ray-project/ray#63814) gives
      map operators a default logical memory footprint so the scheduler
      back-pressures before triggering object-store spills or worker OOM
      kills in IO / transcode / write stages.
    - ``retried_map_errors`` + ``max_map_retries`` (PR ray-project/ray#63023)
      retry map functions when the raised exception looks like a transient
      network / S3 / OS error, up to ``_MAP_MAX_RETRIES`` attempts. A recipe
      that turns its own IO errors into per-item failure rows never reaches
      this layer for those stages; what it backstops there is the stages that
      cannot -- publication writes, where a transient failure has no row to
      become and would otherwise fail the run.
    - ``disable_high_memory_detector`` suppresses reservation-based warnings
      for pipelines that intentionally hold large payloads in memory. The
      detector keeps its existing Ray configuration by default.
    - ``no_progress_timeout_s`` raises (or with ``-1`` removes) the window Ray
      Data waits for one task to finish before failing the execution. Ray's
      default is 30 minutes, which reads a slow task as a wedged one; a recipe
      whose single task legitimately runs longer than that -- one whose map
      function streams gigabytes per item, say -- must widen it or lose the
      whole run to its slowest item. Left alone by default, because a recipe
      with short tasks wants the deadlock detection at Ray's window and not
      hours later.

    Attributes are guarded with ``hasattr`` so importing this module on a
    stale (pre-2.56) Ray wheel still works.
    """
    ctx = ray.data.DataContext.get_current()
    if hasattr(ctx, "default_map_logical_memory_enabled"):
        ctx.default_map_logical_memory_enabled = True  # type: ignore[attr-defined]
    if no_progress_timeout_s is not None and hasattr(ctx, "execution_no_progress_timeout_s"):
        ctx.execution_no_progress_timeout_s = no_progress_timeout_s  # type: ignore[attr-defined]
    # retried_map_errors and max_map_retries both shipped in ray-project/ray#63023;
    # gate them together so a partial-implementation Ray build can't end up with
    # the error list set but no retry cap (or vice versa).
    if hasattr(ctx, "retried_map_errors") and hasattr(ctx, "max_map_retries"):
        ctx.retried_map_errors = list(_MAP_RETRY_PATTERNS)  # type: ignore[attr-defined]
        ctx.max_map_retries = _MAP_MAX_RETRIES  # type: ignore[attr-defined]
    if disable_high_memory_detector:
        issue_detectors_config = getattr(ctx, "issue_detectors_config", None)
        high_memory_detector_config = getattr(issue_detectors_config, "high_memory_detector_config", None)
        if high_memory_detector_config is not None:
            high_memory_detector_config.detection_time_interval_s = -1


def configure_ray_data_eager_actor_autoscaling() -> None:
    """Make expensive actor pools claim newly available resources promptly.

    Ray 2.58 defaults to a 1.75 utilization threshold and adds at most one actor
    per autoscaling decision. That is intentionally conservative, but it
    serializes much of the startup cost for model-loading GPU actors. A 1.0
    threshold and an uncapped step let a saturated pool grow geometrically;
    the operator's maximum size and Ray's resource budget still bound it.
    """
    ctx = ray.data.DataContext.get_current()
    ctx.autoscaling_config.actor_pool_util_upscaling_threshold = 1.0
    ctx.autoscaling_config.actor_pool_max_upscaling_delta = None


def curator_io_slots_per_node() -> int:
    """Return the configured logical IO capacity for a Curator Ray node."""
    raw_value = os.environ.get(
        environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR,
        str(environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE),
    )
    try:
        slots = int(raw_value)
    except ValueError as exc:
        msg = f"{environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR} must be an integer, got {raw_value!r}"
        raise ValueError(msg) from exc
    if slots < 1:
        msg = f"{environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR} must be at least 1, got {slots}"
        raise ValueError(msg)
    return slots


def curator_io_resources() -> dict[str, float]:
    """Return the standard custom resources for a Curator-owned Ray node."""
    return {environment.CURATOR_IO_RESOURCE_NAME: float(curator_io_slots_per_node())}


def ensure_ray_initialized(*, local_resources: Mapping[str, float] | None = None) -> None:
    """Initialize Ray and verify resources required by the calling recipe.

    ``local_resources`` are supplied only when this process owns a new local
    Ray node. Slurm and externally managed clusters must advertise the same
    resources when their nodes start; Ray cannot add them after the fact.
    """
    if not ray.is_initialized():
        connects_to_existing_cluster = bool(os.environ.get("RAY_ADDRESS")) or (
            environment.SLURM_RAY_ENV_VAR_NAME in os.environ
        )
        if local_resources and not connects_to_existing_cluster:
            ray.init(ignore_reinit_error=True, resources=dict(local_resources))
        else:
            ray.init(ignore_reinit_error=True)

    if local_resources:
        cluster_resources = ray.cluster_resources()  # type: ignore[no-untyped-call]
        missing = [name for name in local_resources if float(cluster_resources.get(name, 0.0)) <= 0.0]
        if missing:
            names = ", ".join(sorted(missing))
            msg = (
                f"The connected Ray cluster does not advertise required resource(s): {names}. "
                "Start each Curator work node with the matching custom Ray resources."
            )
            raise RuntimeError(msg)
