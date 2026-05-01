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

"""Demo the feature to dynamically split a pipeline task into multiple tasks for downstream."""

import time

import attrs
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
from cosmos_curator.models.gpt2 import GPT2


# pipeline task object that is being passed between stages
@attrs.define
class DemoTaskChunkingTask(PipelineTask):
    """A pipeline task that processes text prompts and stores generated output."""

    prompt: str
    output: str | None = None


class _LowerCaseStage(CuratorStage):
    def __init__(self) -> None:
        pass

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    def process_data(self, tasks: list[DemoTaskChunkingTask]) -> list[DemoTaskChunkingTask] | None:  # type: ignore[override]
        # convert the prompt to lowercase
        for task in tasks:
            task.prompt = task.prompt.lower()
        # mimic a 200-sec process time
        time.sleep(199)
        return tasks


class _SplitStage(CuratorStage):
    def __init__(self) -> None:
        pass

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=0.5, gpus=0.0)

    def process_data(self, tasks: list[DemoTaskChunkingTask]) -> list[DemoTaskChunkingTask] | None:  # type: ignore[override]
        downstream_tasks = []
        # split the input prompt into individual words
        for task in tasks:
            for word in task.prompt.split():
                new_task = DemoTaskChunkingTask(prompt=word)
                downstream_tasks.append(new_task)
        logger.info(f"Splitting {len(tasks)} input tasks into {len(downstream_tasks)} downstream tasks")
        # mimic a 50-sec process time
        time.sleep(49)
        return downstream_tasks


class _GPT2Stage(CuratorStage):
    def __init__(self) -> None:
        self._model = GPT2()

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.1)

    @property
    def model(self) -> ModelInterface | None:
        return self._model

    def stage_setup(self) -> None:
        self._model.setup()

    def process_data(self, tasks: list[DemoTaskChunkingTask]) -> list[DemoTaskChunkingTask] | None:  # type: ignore[override]
        start_time = time.time()
        for task in tasks:
            # generate text based on the prompt and print out result
            task.output = self._model.generate(task.prompt)
            logger.info(" ".join(task.output.split()))
        # mimic a 10-sec process time
        while time.time() - start_time < 9.9:  # noqa: PLR2004
            time.sleep(0.1)
        return tasks


def main() -> None:
    """Run the demo-task-chunking pipeline."""
    # build a list of input tasks
    prompts = ["hello world hello four", "one", "dummy two"]
    tasks: list[PipelineTask] = [DemoTaskChunkingTask(prompt=x) for x in prompts]
    logger.info(f"Number of input tasks: {len(tasks)}")

    # construct a list of pipeline stages
    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(_LowerCaseStage(), num_workers_per_node=1),
        _GPT2Stage(),
        _SplitStage(),
        _GPT2Stage(),
    ]

    # run the pipeline
    run_pipeline(tasks, stages)

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
