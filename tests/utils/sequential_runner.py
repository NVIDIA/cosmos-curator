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
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility to run a sequence of pipeline stages in tests."""

from collections.abc import Sequence
from typing import TypeVar

from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec, PipelineTask

T = TypeVar("T", bound=PipelineTask)


class SequentialRunner(RunnerInterface):
    """Runner that executes stages sequentially without Ray or distributed computing.

    This runner is primarily used for testing, where we want to run stages
    in a simple sequential manner without the overhead of Ray clusters.
    """

    def run(
        self,
        input_tasks: list[T],
        stage_specs: Sequence[CuratorStageSpec],
        _model_weights_prefix: str,
        _execution_mode: str = "AUTO",
    ) -> list[T] | None:
        """Execute stages sequentially.

        Args:
            input_tasks: A list of pipeline tasks to process.
            stage_specs: A list of stage specifications.
            _model_weights_prefix: Model weights prefix (unused in sequential execution).
            _execution_mode: Execution mode (unused in sequential execution).

        Returns:
            A list of output pipeline tasks.

        """
        # Extract the actual stages from the stage specs
        stages: list[CuratorStage] = [spec.stage for spec in stage_specs]  # type: ignore[untyped-decorator]

        # Setup all stages
        for stage in stages:
            stage.stage_setup()

        # Process tasks through each stage sequentially
        tasks: list[PipelineTask] = input_tasks  # type: ignore[assignment]
        for stage in stages:
            result = stage.process_data(tasks)
            if result is None:
                return None
            tasks = result
            stage.destroy()

        return tasks  # type: ignore[return-value]
