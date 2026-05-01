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

"""Ray cluster lifecycle utilities.

Provides helpers for initialising and shutting down the Ray cluster,
querying live nodes, and registering pre-shutdown hooks.

Pre-shutdown hooks
~~~~~~~~~~~~~~~~~~

Any subsystem that needs to perform work while the Ray cluster is
still alive (e.g. collecting artifacts from worker nodes) can
register a callback via :func:`register_pre_shutdown_hook`.  All
registered hooks run in LIFO order inside :func:`shutdown_cluster`
**before** ``ray.shutdown()`` is called.

::

    register_pre_shutdown_hook(my_cleanup)
    ...
    shutdown_cluster()   # runs my_cleanup(), THEN ray.shutdown()
"""

import os
import socket
import time
from collections.abc import Callable
from typing import Any

import loguru
import ray
from loguru import logger

# Pre-shutdown hooks run in LIFO order before ray.shutdown().
# Module-level list -- populated via register_pre_shutdown_hook().
# Return type is ``object`` so hooks can return any value (discarded).
_pre_shutdown_hooks: list[Callable[[], object]] = []


def _logger_custom_serializer(
    _: "loguru.Logger",
) -> None:
    return None


def _logger_custom_deserializer(
    _: None,
) -> "loguru.Logger":
    # Initialize a default logger
    return logger


def init_or_connect_to_cluster() -> None:
    """Initialize a new local Ray cluster or connects to an existing one.

    If the ``XENNA_RAY_TRACING_HOOK`` environment variable is
    set (by :func:`~cosmos_curator.core.utils.infra.tracing_hook.enable_tracing`),
    the value is forwarded to ``ray.init(_tracing_startup_hook=...)``
    so that every worker runs the OTel tracing setup on startup.
    """
    # Turn off serization for loguru. This is needed as loguru is not serializable in general.
    ray.util.register_serializer(
        logger.__class__,
        serializer=_logger_custom_serializer,
        deserializer=_logger_custom_deserializer,
    )

    tracing_hook = os.environ.get("XENNA_RAY_TRACING_HOOK")
    tracing_kwarg: dict[str, Any] = {"_tracing_startup_hook": tracing_hook} if tracing_hook else {}
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        **tracing_kwarg,
    )


def register_pre_shutdown_hook(hook: Callable[[], object]) -> None:
    """Register a callback to run before ``ray.shutdown()``.

    Hooks are executed in LIFO (last-registered-first) order by
    :func:`shutdown_cluster`.  Each hook runs inside a
    ``contextlib.suppress(Exception)`` guard so that one failing
    hook does not prevent subsequent hooks or the shutdown itself.

    This is the mechanism used by ``ArtifactDelivery`` to collect
    artifacts from worker nodes while Ray is still alive.

    Args:
        hook: Zero-argument callable to execute before shutdown.
            Return value (if any) is discarded.

    """
    hook_name = getattr(hook, "__qualname__", None) or getattr(hook, "__name__", None) or repr(hook)
    logger.debug(f"[ray] Registered pre-shutdown hook: {hook_name}")
    _pre_shutdown_hooks.append(hook)


def shutdown_cluster(*, flush_seconds: float = 1.0) -> None:
    """Run pre-shutdown hooks, flush logs, and cleanly shutdown Ray.

    Pre-shutdown hooks registered via
    :func:`register_pre_shutdown_hook` are executed in LIFO order
    **before** ``ray.shutdown()`` so that they can still communicate
    with the Ray cluster (e.g. deploy actors, call ``ray.get()``).

    ::

        shutdown_cluster()
        |
        +-- 1. Run pre-shutdown hooks (LIFO, suppress errors)
        +-- 2. sleep(flush_seconds)
        +-- 3. ray.shutdown()

    Args:
        flush_seconds: Seconds to sleep before shutdown to allow
            log buffers to drain.  Defaults to 1.0.

    """
    # Run hooks in LIFO order while Ray is still alive.
    if _pre_shutdown_hooks:
        logger.info(f"[ray] Running {len(_pre_shutdown_hooks)} pre-shutdown hook(s)")
    while _pre_shutdown_hooks:
        hook = _pre_shutdown_hooks.pop()
        hook_name = getattr(hook, "__qualname__", None) or getattr(hook, "__name__", None) or repr(hook)
        logger.info(f"[ray] Executing pre-shutdown hook: {hook_name}")
        try:
            hook()
        except Exception:  # noqa: BLE001
            logger.warning(f"[ray] Pre-shutdown hook failed: {hook_name}", exc_info=True)

    time.sleep(flush_seconds)
    ray.shutdown()


def get_live_nodes(*, dump_info: bool = True) -> list[dict[str, str]]:
    """Query the Ray GCS for alive nodes and return their identifiers.

    Calls ``ray.nodes()`` to fetch the full node table from the Ray
    Global Control Store (GCS), then filters to nodes with
    ``Alive=True``.  This is a **read-only metadata query** -- no
    actors are created and no work is dispatched.

    ::

        ray.nodes()  (GCS query, returns all known nodes)
              |
              v
        for each node:
          Alive?
            YES -> extract {NodeID, NodeName} -> live_nodes
            NO  -> log warning (if dump_info), skip

    The returned dicts contain exactly two keys:

    - ``"NodeID"``: opaque Ray-internal hex identifier used for
      scheduling strategies (e.g. ``NodeAffinitySchedulingStrategy``).
    - ``"NodeName"``: the ``NodeManagerHostname`` (usually the
      machine hostname), used for human-readable log messages.

    Args:
        dump_info: If ``True``, logs each discovered node at
            INFO level and warns about dead nodes.  Defaults
            to ``True``.

    Returns:
        List of ``{"NodeID": ..., "NodeName": ...}`` dicts for
        each alive node.  Empty list if no nodes are alive.

    Raises:
        RuntimeError: If ``ray.nodes()`` itself fails (e.g. Ray
            is not initialised or the GCS is unreachable).

    """
    try:
        all_nodes = ray.nodes()  # type: ignore[no-untyped-call]
    except Exception as e:
        logger.error(f"Failed to get nodes: {e}")
        msg = "Failed to get nodes from Ray cluster"
        raise RuntimeError(msg) from e

    live_nodes = []
    for node in all_nodes:
        if node["Alive"]:
            if dump_info:
                logger.info(f"Found node {node['NodeID']} on {node['NodeManagerHostname']}")
            node_info = {
                "NodeID": node["NodeID"],
                "NodeName": node["NodeManagerHostname"],
            }
            live_nodes.append(node_info)
        elif dump_info:
            logger.warning(f"Found node {node['NodeID']} on {node['NodeManagerHostname']} is NOT alive")
    return live_nodes


def get_node_idx_and_name() -> tuple[int, str]:
    """Get the node index and name."""
    node_id = ray.get_runtime_context().get_node_id()
    nodes = get_live_nodes(dump_info=False)
    for idx, node in enumerate(nodes):
        if node["NodeID"] == node_id:
            return idx, node["NodeName"]
    return -1, socket.gethostname()


def get_node_name() -> str:
    """Get the node name."""
    _, node_name = get_node_idx_and_name()
    return node_name
