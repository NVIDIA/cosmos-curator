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

"""Shared Ray Data runtime configuration for Curator Next recipes.

Every function here mutates the process-global ``ray.data.DataContext``, which is
what makes this one module rather than several: progress output, stability knobs,
eager actor autoscaling, and the job checkpoint are all fields on the same
ambient object, and a caller needs to know that setting one of them affects every
dataset the process later builds.

``installed_checkpoint`` is the only one that is scoped rather than set-and-forget,
because leaking a checkpoint config would silently checkpoint an unrelated later
plan under the wrong id column.
"""

import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final

import pyarrow.fs as pafs
import ray
from loguru import logger
from ray.data.checkpoint import CheckpointConfig

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
    "ProbeError",
]
_MAP_MAX_RETRIES = 3
_CHECKPOINT_CONFIG_LOCK = threading.Lock()

# What the startup snapshot reports per node and in total: the two resources Ray
# knows natively plus the logical IO budget Curator adds. Every Curator stage
# reservation is drawn from these three, so a stage that never gets scheduled is
# explained by one of them.
_SNAPSHOT_RESOURCES: Final[tuple[str, ...]] = ("CPU", "GPU", environment.CURATOR_IO_RESOURCE_NAME)

# Above this many nodes only cluster totals are logged. The per-node breakdown
# exists to expose a HETEROGENEOUS cluster, where a total divided by a node count
# is a fiction; a job large enough to exceed this bound would emit a page of log
# per run, which is how a diagnostic gets filtered out and stops being read.
_SNAPSHOT_BREAKDOWN_MAX_NODES: Final = 16


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
      network / S3 / OS error or inconclusive FFprobe failure, up to
      ``_MAP_MAX_RETRIES`` retries. A recipe
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


@contextmanager
def installed_checkpoint(
    *,
    id_column: str,
    checkpoint_path: str,
    filesystem: pafs.FileSystem | None = None,
) -> Iterator[None]:
    """Install one Ray Data job checkpoint for the enclosed plan and always clear it.

    Two properties of the id column decide whether a resume works at all, and both
    are the CALLER's responsibility - this function only names the column::

        Read --> ... --> terminal write
          ^                    ^
          |                    +-- the id column must SURVIVE into the written
          |                        output; Ray marks a row complete by reading
          |                        the column back off the sink
          +-- the id column must be present on the FIRST operator; the checkpoint
              filter reads it off the Read output, so a column added later in the
              plan is invisible to the skip

    A plan that drops the column anywhere between those two points records no
    progress at all while appearing to work. Ray also plans job checkpointing only
    for a logical DAG that ends in a write, so a ``materialize()`` terminal
    disables the mechanism with nothing but a Python warning.

    Args:
        id_column: Column holding one completed row's checkpoint id. Values are
            the caller's; only its own datasource knows the snapshot they are
            drawn from.
        checkpoint_path: Cluster-accessible directory - object storage or a
            network-mounted path. A node-local path gives each worker a
            checkpoint no other worker can read.
        filesystem: Filesystem the checkpoint is read and written through. Pass
            the same one the sink uses, or a run under a non-ambient credential
            profile writes results it cannot read back.

    Yields:
        Nothing; the context is the installation.

    Raises:
        RuntimeError: If a configuration is already installed, or if the active
            one is not the one installed here.

    """
    installed = CheckpointConfig(
        id_column=id_column,
        checkpoint_path=checkpoint_path,
        # The most dangerous default in this design. Ray's default is True, and it
        # deletes the checkpoint when the DATASET EXECUTION succeeds - which is the
        # end of the inference phase, before anything has been published. Accepting
        # the default would destroy the progress record inside exactly the window
        # this configuration exists to protect.
        delete_checkpoint_on_success=False,
        override_filesystem=filesystem,
    )
    with _CHECKPOINT_CONFIG_LOCK:
        context = ray.data.DataContext.get_current()
        # Process-global ambient state, so the guard and installation are one
        # operation: otherwise two threads can both observe None and replace
        # each other's checkpoint before either plan starts.
        if context.checkpoint_config is not None:
            msg = (
                f"a Ray Data checkpoint configuration is already installed "
                f"(id column {context.checkpoint_config.id_column!r}); refusing to replace it"
            )
            raise RuntimeError(msg)
        context.checkpoint_config = installed
        # ASSERTED rather than assumed: setting the config on a context the
        # execution does not use is a silent no-op that costs a full re-run of a
        # GPU phase to discover.
        active = ray.data.DataContext.get_current().checkpoint_config
        if (
            active is None
            or active.id_column != id_column
            or active.checkpoint_path != checkpoint_path
            or active.filesystem is not installed.filesystem
        ):
            context.checkpoint_config = None
            msg = (
                f"the active Ray Data checkpoint configuration is not the one just installed "
                f"(expected id column {id_column!r} at {checkpoint_path!r}, found "
                f"{active and active.id_column!r} at {active and active.checkpoint_path!r}; "
                f"expected filesystem type {type(installed.filesystem).__name__!r}, found "
                f"{active and type(active.filesystem).__name__!r}); "
                f"refusing to run a phase that would record no progress"
            )
            raise RuntimeError(msg)
    try:
        logger.info(f"Installed Ray Data job checkpoint on {id_column!r} at {checkpoint_path}")
        yield
    finally:
        with _CHECKPOINT_CONFIG_LOCK:
            context.checkpoint_config = None


def curator_io_slots_per_node() -> int:
    """Return the configured logical IO capacity for a Curator Ray node.

    Also reports the resolved value and its provenance at INFO: this is the one
    place the environment override is read, so it is the only layer that can
    still tell a configured budget from an inherited one.
    """
    override = os.environ.get(environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR)
    raw_value = str(environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE) if override is None else override
    try:
        slots = int(raw_value)
    except ValueError as exc:
        msg = f"{environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR} must be an integer, got {raw_value!r}"
        raise ValueError(msg) from exc
    if slots < 1:
        msg = f"{environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR} must be at least 1, got {slots}"
        raise ValueError(msg)
    # Provenance, not just the value. A starved IO budget reads the same either
    # way, but "16 was chosen" and "16 was inherited" imply opposite fixes, and
    # nothing downstream can tell them apart once the number is a plain float on
    # a node. Only what THIS process can see is claimed: sbatch.sh.j2 exports the
    # variable on every Slurm job, from --ray-io-slots-per-node when it is passed
    # and from the same built-in default when it is not, so "set" inside a Slurm
    # job is not evidence that an operator picked the number.
    if override is None:
        provenance = f"the built-in default ({environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR} is unset)"
    else:
        provenance = (
            f"{environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR}={override!r}, which a Slurm job exports "
            f"whether or not --ray-io-slots-per-node was passed"
        )
    logger.info(
        f"Curator node IO budget resolved to {slots} {environment.CURATOR_IO_RESOURCE_NAME} slot(s) "
        f"per node from {provenance}."
    )
    return slots


def curator_io_resources() -> dict[str, float]:
    """Return the standard custom resources for a Curator-owned Ray node."""
    return {environment.CURATOR_IO_RESOURCE_NAME: float(curator_io_slots_per_node())}


def _render_resources(resources: Mapping[str, float]) -> str:
    """Render one node's or one cluster's resource counts in snapshot order."""
    return ", ".join(f"{resources[name]:g} {name}" for name in _SNAPSHOT_RESOURCES)


def _log_cluster_snapshot() -> None:
    """Log the resources the connected Ray cluster advertises right now.

    A SNAPSHOT, and the text says so. Slurm releases the driver as soon as the
    FIRST worker registers (``slurm_ray_runtime._wait_for_first_worker``), so a
    multi-node job routinely reads its own cluster here while later nodes are
    still joining::

        head up --> first worker joins --> DRIVER STARTS --> workers 2..N join
                                              ^
                                              +-- this reading

    Waiting for a complete picture would trade a diagnostic for a startup stall
    on every recipe, so the count is published with its uncertainty stated rather
    than waited into being final.
    """
    nodes = ray.nodes()  # type: ignore[no-untyped-call]  # Missing in Ray stubs.
    advertised = [
        (
            str(node.get("NodeManagerAddress") or node.get("NodeID", "unknown")),
            {name: float(node.get("Resources", {}).get(name, 0.0)) for name in _SNAPSHOT_RESOURCES},
        )
        for node in nodes
        if node.get("Alive", False)
    ]
    totals = {name: sum(resources[name] for _, resources in advertised) for name in _SNAPSHOT_RESOURCES}
    lines = [
        f"Ray cluster snapshot at driver start: {len(advertised)} live node(s) advertising "
        f"{_render_resources(totals)} in total. Nodes still joining are not counted."
    ]
    if len(advertised) <= _SNAPSHOT_BREAKDOWN_MAX_NODES:
        lines.extend(f"  node {address}: {_render_resources(resources)}" for address, resources in advertised)
    logger.info("\n".join(lines))


def ensure_ray_initialized(
    *,
    local_resources: Mapping[str, float] | None = None,
    runtime_env_vars: Mapping[str, str] | None = None,
) -> None:
    """Initialize Ray and verify resources required by the calling recipe.

    ``local_resources`` are supplied only when this process owns a new local
    Ray node. Slurm and externally managed clusters must advertise the same
    resources when their nodes start; Ray cannot add them after the fact.
    ``runtime_env_vars`` configure every task and actor in a newly initialized
    Ray job; they cannot change one that was already initialized. One INFO
    snapshot of what the cluster advertises is logged on the way through, since
    this is the one point every recipe passes and the verification below reads
    the same figures.
    """
    if not ray.is_initialized():
        connects_to_existing_cluster = bool(os.environ.get("RAY_ADDRESS")) or (
            environment.SLURM_RAY_ENV_VAR_NAME in os.environ
        )
        init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
        if runtime_env_vars:
            init_kwargs["runtime_env"] = {"env_vars": dict(runtime_env_vars)}
        if local_resources and not connects_to_existing_cluster:
            init_kwargs["resources"] = dict(local_resources)
        ray.init(**init_kwargs)

    # Before the check below, so the refusal it can raise arrives with the
    # reading that explains it rather than after an operator goes looking.
    _log_cluster_snapshot()

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
