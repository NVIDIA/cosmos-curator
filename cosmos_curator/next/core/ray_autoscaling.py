# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release surplus pending download actors that block Ray 2.58 scale-down.

Ray's default autoscaler refuses utilization-based scale-down while an actor is
pending, even when idle actors hold every curator_io slot. Use the execution
callback hook to remove a pending actor only when ready actors already have spare
capacity. Ray retains control of cooldowns, idle-actor retirement, and scale-up.
This workaround depends on the pinned Ray executor and actor-pool interfaces.
"""

from ray.data._internal.actor_autoscaler.autoscaling_actor_pool import ActorPoolScalingRequest
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment


def configure_download_actor_autoscaling(context: DataContext) -> None:
    """Scope reachable scale-up and pending-actor cleanup to a media plan's context."""
    context.autoscaling_config.actor_pool_util_upscaling_threshold = 1.0
    if ReleasePendingDownloadActors not in context.custom_execution_callback_classes:
        context.custom_execution_callback_classes.append(ReleasePendingDownloadActors)


class ReleasePendingDownloadActors(ExecutionCallback):
    """Unblock normal scale-down for idle pools that reserve curator_io slots."""

    def on_execution_step(self, executor: StreamingExecutor) -> None:
        """Cancel at most one surplus pending actor per pool, respecting Ray's cooldown."""
        for op in executor._topology or ():  # noqa: SLF001 -- pinned Ray execution callback interface
            if not isinstance(op, ActorPoolMapOperator):
                continue
            resources = op._ray_remote_args.get("resources", {})  # noqa: SLF001 -- identify IO pools in pinned Ray
            if resources.get(environment.CURATOR_IO_RESOURCE_NAME, 0) <= 0 or op.metrics.num_inputs_received == 0:
                continue
            threshold = op.data_context.autoscaling_config.actor_pool_util_downscaling_threshold
            for pool in op.get_autoscaling_actor_pools():
                if (
                    pool.num_pending_actors() > 0
                    and pool.num_alive_actors() > pool.num_active_actors()
                    and pool.current_size() > pool.min_size()
                    and pool.get_pool_util() <= threshold
                ):
                    # Ray removes pending actors first; force=False retains the
                    # cooldown after scale-up and leaves active tasks untouched.
                    pool.scale(
                        ActorPoolScalingRequest.downscale(
                            delta=-1, reason="idle download actors have spare capacity while an actor is pending"
                        )
                    )
