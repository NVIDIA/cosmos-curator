# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pending download actors release the autoscaler's guard without interrupting work."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ray.data._internal.actor_autoscaler.autoscaling_actor_pool import (
    ActorPoolScalingRequest,
    AutoscalingActorConfig,
)
from ray.data._internal.actor_autoscaler.default_actor_autoscaler import DefaultActorAutoscaler
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators import actor_pool_map_operator
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator, _ActorPool
from ray.data.context import AutoscalingConfig, DataContext

from cosmos_curator.next.core.ray_autoscaling import (
    ReleasePendingDownloadActors,
    configure_download_actor_autoscaling,
)


def test_download_autoscaling_configuration_is_scoped_and_idempotent() -> None:
    """Repeated configuration preserves other callbacks and never mutates the ambient context."""
    context = DataContext()
    original = list(context.custom_execution_callback_classes)
    scoped = context.copy()
    configure_download_actor_autoscaling(scoped)
    configure_download_actor_autoscaling(scoped)
    assert scoped.autoscaling_config.actor_pool_util_upscaling_threshold == 1.0
    assert scoped.custom_execution_callback_classes == [*original, ReleasePendingDownloadActors]
    assert context.custom_execution_callback_classes == original


def test_pending_actor_cleanup_preserves_cooldown_active_tasks_and_future_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise Ray's real pool and autoscaler with only Core actor creation mocked."""
    now = [0.0]
    monkeypatch.setattr(actor_pool_map_operator, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(actor_pool_map_operator.ray, "get", lambda _: "test-node")
    usage = ExecutionResources(cpu=0.25, memory=512 * 1024**2)
    pool = _ActorPool(
        lambda *_: (Mock(), Mock(), usage),
        AutoscalingActorConfig(
            min_size=1,
            max_size=float("inf"),
            initial_size=1,
            max_tasks_in_flight_per_actor=1,
            max_actor_concurrency=1,
            per_actor_resource_usage=usage,
        ),
    )
    pool.scale(ActorPoolScalingRequest.upscale(delta=17))
    for ref in pool.get_pending_actor_refs()[:16]:
        pool.pending_to_running(ref)
    for _ in range(7):
        pool.on_task_submitted(pool.select_actors())

    op = Mock(spec=ActorPoolMapOperator)
    op._ray_remote_args = {"resources": {"curator_io": 1}}
    op.data_context = DataContext()
    op.metrics.num_inputs_received = 100
    op.get_autoscaling_actor_pools.return_value = [pool]
    op.has_completed.return_value = False
    op._inputs_complete = False
    state = Mock()
    state.total_enqueued_input_blocks.return_value = 100
    state._scheduling_status.under_resource_limits = True
    topology = {op: state}
    resources = Mock()
    resources.get_allocation.return_value = None
    resources.get_op_usage.return_value = None
    resources.get_budget.return_value = None
    autoscaler = DefaultActorAutoscaler(
        topology, resources, config=AutoscalingConfig(actor_pool_util_upscaling_threshold=1.0)
    )
    callback = ReleasePendingDownloadActors()
    executor = Mock(_topology=topology)

    autoscaler.try_trigger_scaling()
    assert pool.current_size() == 17
    callback.on_execution_step(executor)
    assert pool.num_pending_actors() == 1  # The ten-second cooldown still applies.
    now[0] = 11.0
    callback.on_execution_step(executor)
    assert pool.num_pending_actors() == 0
    assert pool.num_running_actors() == 16
    for _ in range(5):
        autoscaler.try_trigger_scaling()
    assert pool.num_running_actors() == 13  # Seven tasks exceed the 0.5 threshold again.
    assert pool.num_active_actors() == pool.num_tasks_in_flight() == 7

    # New demand can grow the pool again, and a late worker can start that actor.
    while pool.num_idle_actors():
        pool.on_task_submitted(pool.select_actors())
    autoscaler.try_trigger_scaling()
    callback.on_execution_step(executor)
    assert pool.num_pending_actors() == 1
    pool.pending_to_running(pool.get_pending_actor_refs()[0])
    assert pool.num_running_actors() == 14


@pytest.mark.parametrize(
    ("alive", "active", "pending", "size", "minimum", "util", "inputs", "io_slots"),
    [
        (0, 0, 3, 3, 1, 0.0, 1, 1),  # Starting or restarting actors are not idle capacity.
        (2, 2, 3, 5, 1, 0.4, 1, 1),  # Pending actors alone lower utilization; ready actors are busy.
        (2, 0, 1, 3, 3, 0.0, 1, 1),  # Do not scale below the configured minimum.
        (3, 2, 1, 4, 1, 0.75, 1, 1),  # Respect the configured utilization threshold.
        (2, 0, 1, 3, 1, 0.0, 0, 1),  # Do not interfere with initial provisioning.
        (2, 0, 1, 3, 1, 0.0, 1, 0),  # Model and other actor pools are outside this workaround.
        (2, 0, 0, 2, 1, 0.0, 1, 1),  # Ray already handles ordinary idle-actor retirement.
    ],
)
def test_pending_actor_cleanup_leaves_needed_capacity_alone(
    alive: int, active: int, pending: int, size: int, minimum: int, util: float, inputs: int, io_slots: int
) -> None:
    """Only an underutilized IO pool with spare ready capacity can cancel a pending actor."""
    pool = Mock()
    pool.num_alive_actors.return_value = alive
    pool.num_active_actors.return_value = active
    pool.num_pending_actors.return_value = pending
    pool.current_size.return_value = size
    pool.min_size.return_value = minimum
    pool.get_pool_util.return_value = util
    op = Mock(spec=ActorPoolMapOperator)
    op._ray_remote_args = {"resources": {"curator_io": io_slots}}
    op.data_context = DataContext()
    op.metrics.num_inputs_received = inputs
    op.get_autoscaling_actor_pools.return_value = [pool]
    ReleasePendingDownloadActors().on_execution_step(Mock(_topology={op: Mock()}))
    pool.scale.assert_not_called()
