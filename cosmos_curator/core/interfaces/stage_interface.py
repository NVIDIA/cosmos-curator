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
"""Key interfaces for defining a stage in the curation pipeline."""

import attrs
import ray

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_xenna.pipelines.private.resources import NodeInfo, Resources, WorkerMetadata
from cosmos_xenna.pipelines.private.specs import Stage, StageSpec
from cosmos_xenna.ray_utils.runtime_envs import CondaEnv, RuntimeEnv


@attrs.define
class PipelineTask:
    """Base class for pipeline task or payload."""

    @property
    def weight(self) -> float:
        """Return the weight of the pipeline task.

        Returns:
            The weight of the pipeline task.

        """
        return 1.0

    @property
    def fraction(self) -> float:
        """Return the fraction of the pipeline task.

        Returns:
            The fraction of the pipeline task.

        """
        return 1.0

    def get_major_size(self) -> int:
        """Return the major size (bytes) of the task for performance accounting.

        Subclasses should override this when they can provide a meaningful size.
        The default implementation returns 0.

        """
        return 0


@attrs.define
class CuratorStageResource:
    """Define resource requirements for a stage."""

    cpus: float = 1.0
    gpus: float | int = 0


class CuratorStage(Stage[PipelineTask, PipelineTask]):
    """Base class for a stage in curation pipeline.

    The very base class Stage[T, V] is templated with T and V, where T is the type of input task
    while V is the type of the output task for this stage.
    Here we use PipelineTask as both the input and output task type because all pipeline task class should inherit
    from PipelineTask class. When building a pipeline, they can be different derived classes.
    """

    @property
    def resources(self) -> CuratorStageResource:
        """Need to override this method to define the resource requirements for the stage."""
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    @property
    def model(self) -> ModelInterface | None:
        """Need to override this method to define the model used in the stage."""
        return None

    @property
    def conda_env_name(self) -> str | None:
        """Need to override this method to define the conda environment name used in the stage.

        By default, it will return the conda env name of the model if it is not None.
        """
        if self.model is not None:
            return self.model.conda_env_name
        return None

    def stage_setup_on_node(self) -> None:
        """Optionally override this method to define the setup process per node.

        This method is called exactly once per node, and is guaranteed to be called before the setup method.
        This is useful if you need to do per node setup, like copying model weights from shared filesystem to local SSD.
        """
        return

    def stage_setup(self) -> None:
        """Need to override this method to define the setup process for the stage.

        One key difference between this method and the constructor is that this setup method
        runs in the remote actor process in the desired conda environment while the constructor
        runs in the main process in default conda environment.
        """
        if self.model is not None:
            self.model.setup()

    def process_data(self, task: list[PipelineTask]) -> list[PipelineTask] | None:
        """Need to override this method to define the processing logic for the stage."""
        return task

    def destroy(self) -> None:
        """Need to override this method to define the destroy process for the stage."""
        return

    # Should not override
    @property
    def stage_batch_size(self) -> int:
        """Specify how many tasks to process in a batch.

        Returns:
            The batch size for the stage.

        """
        return 1

    @property
    def required_resources(self) -> Resources:
        """Return the required resources for the stage.

        Returns:
            The required resources for the stage.

        """
        return Resources(
            cpus=self.resources.cpus,
            gpus=self.resources.gpus,
        )

    # Should not override
    @property
    def env_info(self) -> RuntimeEnv | None:
        """Return the environment information for the stage.

        Returns:
            The environment information for the stage.

        """

        class _PixiRuntimeEnv(RuntimeEnv):
            def to_ray_runtime_env(self) -> ray.runtime_env.RuntimeEnv:
                return PixiRuntimeEnv(
                    self.conda.name if self.conda else "",
                    env_vars=self.extra_env_vars,
                )

        if self.conda_env_name is not None:
            return _PixiRuntimeEnv(CondaEnv(self.conda_env_name))
        return None

    # Should not override
    def setup_on_node(self, _node_info: NodeInfo, _worker_metadata: WorkerMetadata) -> None:
        """Set up the stage on a specific node.

        Args:
            _node_info: The node information.
            _worker_metadata: The worker metadata.

        """
        self.stage_setup_on_node()

    # Should not override
    def setup(self, _: WorkerMetadata) -> None:
        """Set up the stage.

        Args:
            _: The worker metadata.

        """
        self.stage_setup()


@attrs.define
class CuratorStageSpec(StageSpec[PipelineTask, PipelineTask]):
    """Define additional properties of a pipeline stage.

    Commonly used properties are:
    num_workers_per_node: set a fixed number of workers per node, typically for IO workers.
    num_run_attempts_python: set the number of retry attempts in case of exceptions.
    over_provision_factor: set the over-provision factor for the stage to influence stage worker allocation.
    worker_max_lifetime_m: set the maximum lifetime of a worker in minutes.
    worker_restart_interval_m: set the interval in minutes for two lifetime-expiring worker restarts.
    """

    def display_str(self) -> str:
        """Return the display string for the stage.

        Returns:
            The display string for the stage.

        """
        res: str = self.name()
        res += f" num_workers_per_node={self.num_workers_per_node}"
        res += f" cpus={self.stage.required_resources.cpus}"
        res += f" gpus={self.stage.required_resources.gpus}"
        return res
