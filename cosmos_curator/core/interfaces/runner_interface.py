# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Runner interface for executing pipelines with different backends."""

import abc
from typing import TypeVar

import attrs
import ray
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec, PipelineTask
from cosmos_curator.core.utils.infra import ray_cluster_utils
from cosmos_xenna.pipelines.private.pipelines import run_pipeline as xenna_run_pipeline
from cosmos_xenna.pipelines.private.specs import (
    PipelineConfig,
    PipelineSpec,
    StreamingSpecificSpec,
)
from cosmos_xenna.utils.verbosity import VerbosityLevel

T = TypeVar("T", bound=PipelineTask)


class RunnerInterface(abc.ABC):
    """Abstract base class that defines an interface for pipeline runners.

    A runner is responsible for executing a pipeline with a specific backend or execution strategy.
    Different implementations can provide alternative ways of running pipelines beyond the default
    Xenna-based execution.
    """

    @abc.abstractmethod
    def run(
        self,
        input_tasks: list[T],
        stage_specs: list[CuratorStageSpec],
        model_weights_prefix: str,
        execution_mode: str = "AUTO",
    ) -> list[T] | None:
        """Execute the pipeline with the given configuration.

        Args:
            input_tasks: A list of pipeline tasks to process.
            stage_specs: A list of stage specifications with filled-in defaults.
            model_weights_prefix: Prefix for model weights in local or cloud storage.
            execution_mode: ``"AUTO"`` (default) selects the mode automatically based on available
                GPUs; ``"STREAMING"`` or ``"BATCH"`` forces the respective mode.

        Returns:
            A list of output pipeline tasks, or None if the pipeline produces no output.

        """


class XennaRunner(RunnerInterface):
    """Default runner implementation using the Cosmos-Xenna framework.

    This runner executes pipelines using the Xenna streaming/batch execution engine
    with Ray for distributed computing.

    Per-pipeline tuning of Xenna's streaming-mode behavior (autoscaler windows, queue
    backpressure, scale-down policy) is supported via the ``streaming_spec`` constructor
    argument or the :meth:`with_streaming_overrides` convenience classmethod. When omitted,
    the spec returned by :meth:`_default_streaming_spec` is used.
    """

    def __init__(self, streaming_spec: StreamingSpecificSpec | None = None) -> None:
        """Initialize the runner with an optional streaming-mode spec.

        Args:
            streaming_spec: Xenna ``StreamingSpecificSpec`` to apply when execution
                mode resolves to STREAMING. ``None`` means use
                :meth:`_default_streaming_spec`. Ignored by Xenna in BATCH mode.

        """
        self._streaming_spec: StreamingSpecificSpec = streaming_spec or self._default_streaming_spec()

    @staticmethod
    def _default_streaming_spec() -> StreamingSpecificSpec:
        """Return the default ``StreamingSpecificSpec`` shared by all pipelines."""
        return StreamingSpecificSpec(
            autoscale_interval_s=60 * 3.0,
            autoscale_speed_estimation_window_duration_s=60 * 3.0,
            max_queued_multiplier=1.5,
            max_queued_lower_bound=16,
            autoscaler_verbosity_level=VerbosityLevel.NONE,
            executor_verbosity_level=VerbosityLevel.NONE,
        )

    @classmethod
    def with_streaming_overrides(cls, **overrides: object) -> "XennaRunner":
        """Build a runner whose default streaming spec is evolved with ``overrides``.

        Pipelines that need to tune a small number of streaming-mode knobs can call
        this without having to repeat all of the unchanged defaults from
        :meth:`_default_streaming_spec`. ``overrides`` are forwarded to
        :func:`attrs.evolve`, so any field on
        :class:`cosmos_xenna.pipelines.private.specs.StreamingSpecificSpec` is valid.

        Args:
            **overrides: Field/value pairs to override on the default spec.

        Returns:
            A new :class:`XennaRunner` whose ``streaming_spec`` is the default spec
            evolved with ``overrides``.

        """
        spec = attrs.evolve(cls._default_streaming_spec(), **overrides)  # type: ignore[arg-type]
        return cls(streaming_spec=spec)

    def run(
        self,
        input_tasks: list[T],
        stage_specs: list[CuratorStageSpec],
        model_weights_prefix: str,
        execution_mode: str = "AUTO",
    ) -> list[T] | None:
        """Execute the pipeline using Cosmos-Xenna.

        Args:
            input_tasks: A list of pipeline tasks to process.
            stage_specs: A list of stage specifications with filled-in defaults.
            model_weights_prefix: Prefix for model weights in local or cloud storage.
            execution_mode: ``"AUTO"`` (default) selects the mode automatically based on available
                GPUs; ``"STREAMING"`` or ``"BATCH"`` forces the respective mode.

        Returns:
            A list of output pipeline tasks, or None if the pipeline produces no output.

        """
        # Import here to avoid circular dependencies
        from cosmos_curator.core.interfaces.pipeline_interface import _prepare_to_run_pipeline  # noqa: PLC0415
        from cosmos_curator.core.utils.config.operation_context import (  # noqa: PLC0415
            is_running_on_slurm,
            is_running_on_the_cloud,
        )

        # Prepare the pipeline (download models, determine execution mode)
        resolved_execution_mode = _prepare_to_run_pipeline(stage_specs, model_weights_prefix, execution_mode)

        # Construct the pipeline configuration
        pipeline_config = PipelineConfig(
            execution_mode=resolved_execution_mode,
            enable_work_stealing=False,
            return_last_stage_outputs=True,
            actor_pool_verbosity_level=VerbosityLevel.NONE,
            monitoring_verbosity_level=VerbosityLevel.NONE
            if is_running_on_the_cloud() and not is_running_on_slurm()
            else VerbosityLevel.INFO,
            mode_specific=self._streaming_spec,
        )

        try:
            logger.info(
                f"Running pipeline in {pipeline_config.execution_mode.name} mode with {len(input_tasks)} input tasks"
            )
            pipeline_spec = PipelineSpec(input_data=input_tasks, stages=stage_specs, config=pipeline_config)
            output_tasks = xenna_run_pipeline(pipeline_spec)

            if output_tasks is None:
                logger.warning("Pipeline execution returned None")
                return None
            return output_tasks
        finally:
            # Only shutdown Ray if it was initialized (may have been initialized by this runner or elsewhere)
            if ray.is_initialized():
                ray_shutdown_delay = 5
                logger.info(f"Disconnecting from Ray cluster in {ray_shutdown_delay} seconds")
                ray_cluster_utils.shutdown_cluster(flush_seconds=ray_shutdown_delay)
