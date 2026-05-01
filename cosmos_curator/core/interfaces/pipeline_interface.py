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
"""Entry function to run a pipeline."""

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import ray
from loguru import logger

from cosmos_curator.core.interfaces.runner_interface import RunnerInterface, XennaRunner
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec, PipelineTask
from cosmos_curator.core.utils.config.operation_context import get_tmp_dir
from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curator.core.utils.infra import hardware_info, ray_cluster_utils
from cosmos_curator.core.utils.infra.profiling import _apply_profiling_config, profiling_wrapper
from cosmos_curator.core.utils.misc.stage_replay import StageSaveConfig, should_save_stage, stage_save_wrapper
from cosmos_curator.core.utils.model.model_utils import download_model_weights_on_all_nodes
from cosmos_xenna.pipelines.private.specs import (
    ExecutionMode,
)


class PipelineExecutionError(Exception):
    """Exception raised when pipeline execution fails.

    This exception is used to wrap and provide context for various types of failures
    that can occur during pipeline execution, such as:
    - Invalid execution mode
    - Stage execution failures
    - Resource allocation failures
    - Model download failures

    Attributes:
        message: A human-readable error message describing the failure.
        original_error: The original exception that caused the pipeline failure, if any.

    """

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize PipelineExecutionError.

        Args:
            message: A human-readable error message describing the failure.
            original_error: The original exception that caused the pipeline failure, if any.

        """
        self.message = message
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        """Return a string representation of the error.

        Returns:
            A string containing both the error message and the original error if present.

        """
        if self.original_error:
            return f"{self.message} (Original error: {self.original_error!s})"
        return self.message


def _worker_download_models(model_names: list[str], model_weights_prefix: str) -> int:
    """Worker function to prepare the pipeline by downloading models.

    Args:
        model_names: List of model names to download.
        model_weights_prefix: Prefix for model weights in local or cloud storage.

    Returns:
        int: Number of GPUs available in the Ray cluster.

    """
    ray_cluster_utils.init_or_connect_to_cluster()
    num_gpus: int = ray.cluster_resources().get("GPU", 0)  # type: ignore[no-untyped-call]
    num_nodes = len(ray_cluster_utils.get_live_nodes())
    logger.info(f"The ray cluster has {num_gpus} GPUs on {num_nodes} nodes.")

    # schedule a model-downloader task on each node
    download_model_weights_on_all_nodes(model_names, model_weights_prefix, num_nodes)
    # dump disk usage info
    hardware_info.print_disk_path_info(get_tmp_dir())

    ray_cluster_utils.shutdown_cluster()
    return num_gpus


def download_models(model_names: list[str], model_weights_prefix: str = MODEL_WEIGHTS_PREFIX) -> int:
    """Download model weights on all nodes in the Ray cluster.

    Args:
        model_names: List of model names to download.
        model_weights_prefix: Prefix for model weights in local or cloud storage.

    Returns:
        int: Number of GPUs available in the Ray cluster.

    """
    # start a separate process to keep main process clean
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker_download_models, list(model_names), model_weights_prefix)
        return future.result()


def _prepare_to_run_pipeline(
    stages: list[CuratorStageSpec],
    model_weights_prefix: str = MODEL_WEIGHTS_PREFIX,
    execution_mode: str = "AUTO",
) -> ExecutionMode:
    """Run a few steps to prepare the pipeline for execution.

    - downloading required models
    - determining execution mode

    Args:
        stages: List of pipeline stages to process.
        model_weights_prefix: Prefix for model weights in local or cloud storage.
        execution_mode: ``"AUTO"`` selects STREAMING when enough GPUs are available, otherwise
            BATCH. ``"STREAMING"`` or ``"BATCH"`` forces the respective mode.

    Returns:
        ExecutionMode: The execution mode for the pipeline.

    """
    model_names: set[str] = set()
    num_gpus_requested: float = 0
    logger.debug(f"Number of pipeline stages: {len(stages)}")
    for stage in stages:
        # mypy type-narrowing
        if stage.stage.model is not None:  # type: ignore[attr-defined]
            model_names.update(stage.stage.model.model_id_names)  # type: ignore[attr-defined]
        num_gpus_requested += stage.stage.required_resources.gpus
        logger.debug(stage.display_str())

    num_gpus_available = download_models(list(model_names), model_weights_prefix)
    logger.info(
        f"Required GPUs to run STREAMING mode vs. available GPUs: {num_gpus_requested} vs. {num_gpus_available}"
    )

    if execution_mode == "AUTO":
        return ExecutionMode["STREAMING"] if num_gpus_requested <= num_gpus_available else ExecutionMode["BATCH"]

    mode = ExecutionMode[execution_mode]
    if mode is ExecutionMode.STREAMING and num_gpus_requested > num_gpus_available:
        logger.warning(
            f"STREAMING mode requested but stages want {num_gpus_requested} GPUs and only "
            f"{num_gpus_available} are available; pipeline may fail to schedule."
        )
    return mode


def _conditionally_wrap_stage(
    stage: CuratorStage | CuratorStageSpec, stage_save_config: StageSaveConfig | None
) -> CuratorStage | CuratorStageSpec:
    """Wrap the stage with the stage save wrapper if the stage should be saved.

    Args:
        stage: The stage to wrap.
        stage_save_config: The configuration for saving stages.

    Returns:
        The stage or stage spec with the process_data method wrapped.

    """
    if stage_save_config is None:
        return stage
    if should_save_stage(stage, stage_save_config):
        return stage_save_wrapper(stage, stage_save_config)
    return stage


def _fill_stage_spec_defaults(stage_specs: list[CuratorStageSpec]) -> None:
    """Fill in default lifetime and restart configuration for stage specs.

    Sets ``worker_max_lifetime_m`` and ``worker_restart_interval_m``
    when they are ``None``, using heuristics based on the stage's
    resource requirements:

    * GPU stages are more expensive to restart -- longer lifetime,
      longer restart interval.
    * Sub-CPU stages (< 1.0 CPUs) are likely I/O stages -- no
      automatic restart.
    * CPU stages get moderate defaults.

    Args:
        stage_specs: Stage specs to mutate in place.

    """
    for spec in stage_specs:
        if spec.worker_max_lifetime_m is None:
            if spec.stage.required_resources.gpus > 0.0:
                # GPU stage can be more expensive to restart
                spec.worker_max_lifetime_m = 120
            elif spec.stage.required_resources.cpus < 1.0:
                # likely an IO stage, do not restart
                spec.worker_max_lifetime_m = 0
            else:
                spec.worker_max_lifetime_m = 60
        if spec.worker_restart_interval_m is None:
            if spec.stage.required_resources.gpus > 0.0:
                # again GPU stage can be more expensive to restart
                spec.worker_restart_interval_m = 5
            else:
                spec.worker_restart_interval_m = 1


def _build_pipeline_stage_specs(
    stages: Sequence[CuratorStage | CuratorStageSpec],
    stage_save_config: StageSaveConfig | None,
    args: argparse.Namespace | None = None,
) -> list[CuratorStageSpec]:
    """Build a list of pipeline stage specs from the given stages.

    Normalises a mixed sequence of ``CuratorStage`` and
    ``CuratorStageSpec`` into a uniform ``list[CuratorStageSpec]``,
    applies optional wrappers (task saving, instrumentation), and
    fills in default lifetime / restart configuration.

    Args:
        stages: Heterogeneous sequence of bare stages or fully
            specified stage specs.  Bare stages are wrapped in a
            default ``CuratorStageSpec``.
        stage_save_config: When provided, stages whose names match
            the config are wrapped with ``stage_save_wrapper()`` so
            that input tasks are persisted to disk for later replay.
        args: Parsed CLI namespace.  When provided,
            ``_apply_profiling_config()`` is called to derive a
            ``ProfilingConfig`` and every stage is transparently
            wrapped with the requested instrumentation backends via
            ``profiling_wrapper()``.  Profiling artifacts are
            written to ``<output-path>/profile``.

    Returns:
        A list of ``CuratorStageSpec`` ready for execution.

    Raises:
        PipelineExecutionError: If any element of *stages* is neither
            a ``CuratorStage`` nor a ``CuratorStageSpec``.

    """
    # Unify the stage spec and stage
    stage_specs: list[CuratorStageSpec] = []
    for stage in stages:
        _stage = _conditionally_wrap_stage(stage, stage_save_config)
        if isinstance(_stage, CuratorStage):
            stage_specs.append(CuratorStageSpec(_stage))
        elif isinstance(_stage, CuratorStageSpec):
            stage_specs.append(_stage)
        else:
            err_msg = f"Invalid stage type: {type(_stage)}. Expected CuratorStage or CuratorStageSpec."  # type: ignore[unreachable]
            raise PipelineExecutionError(err_msg, original_error=None)

    # Apply instrumentation wrapper to every stage (transparent, automatic).
    profiling_config = _apply_profiling_config(args) if args is not None else None
    if profiling_config is not None:
        for spec in stage_specs:
            spec.stage = profiling_wrapper(spec.stage, profiling_config)  # type: ignore[arg-type]

    _fill_stage_spec_defaults(stage_specs)
    return stage_specs


T = TypeVar("T", bound=PipelineTask)


def run_pipeline[T: PipelineTask](  # noqa: PLR0913
    input_tasks: list[T],
    stages: Sequence[CuratorStage | CuratorStageSpec],
    model_weights_prefix: str = MODEL_WEIGHTS_PREFIX,
    runner: RunnerInterface | None = None,
    stage_save_config: StageSaveConfig | None = None,
    args: argparse.Namespace | None = None,
) -> list[T]:
    """Run the pipeline with the given pipeline spec.

    Args:
        input_tasks: A list of pipeline tasks to process.
        stages: A list of stages.
        model_weights_prefix: Prefix for model weights in local or cloud storage.
        runner: Runner implementation for executing the pipeline. Defaults to XennaRunner.
        stage_save_config: Configuration for saving stages for replay.
        args: Parsed CLI namespace.  When provided,
            ``_apply_profiling_config(args)`` is called internally
            and every stage is automatically wrapped with the
            enabled backends (e.g. pyinstrument, memray,
            torch.profiler).  Profiling artifacts are written to
            ``<output-path>/profile``.

    Returns:
        A list of pipeline payloads.

    Raises:
        PipelineExecutionError: If pipeline execution fails.

    """
    if runner is None:
        runner = XennaRunner()

    # Build a list of StageSpecs and fill in default config values.
    stage_specs = _build_pipeline_stage_specs(stages, stage_save_config, args)
    execution_mode = getattr(args, "execution_mode", "AUTO") if args is not None else "AUTO"

    # Run the pipeline!
    try:
        output_tasks = runner.run(input_tasks, stage_specs, model_weights_prefix, execution_mode)
    except Exception as e:
        err_msg = f"Pipeline execution failed: {e!s}"
        logger.error("Pipeline execution failed: ")
        raise PipelineExecutionError(err_msg, original_error=e) from e
    else:
        if output_tasks is None:
            logger.warning("Pipeline execution returned None")
            return []
        return output_tasks
