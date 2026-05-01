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
"""Stage replay.

Implementation notes:

To improve testability, protocol classes are defined for dependency injection.

This leads to a longer implementation, but it makes the code more testable.

The code is organized into the following sections:

- Protocols for dependency injection
- Default implementations of the protocols
- Helper functions
- Public API
"""

import argparse
import fnmatch
import pickle
import random
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from urllib.parse import urlparse

import attrs
import ray
import smart_open  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec, PipelineTask
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.pipelines.video.utils.data_model import Video
from cosmos_xenna.pipelines.private.resources import NodeInfo, WorkerMetadata

BaseStage = TypeVar("BaseStage", bound="CuratorStage")
TaskPath = Path | storage_client.StoragePrefix

MAX_STAGE_REPLAY_ARGS = 2
MAX_STAGE_COMPARE_ARGS = 2


# ============================================================================
# Protocols for Dependency Injection (for testability)
# ============================================================================


class TaskSerializer(Protocol):
    """Protocol for task serialization and deserialization."""

    def save(self, path: TaskPath, tasks: list[PipelineTask]) -> None:
        """Save tasks to a file.

        Args:
            path: Path to save the tasks to.
            tasks: Tasks to save.

        """
        ...  # pragma: no cover

    def load(self, path: TaskPath) -> list[PipelineTask]:
        """Load tasks from a file.

        Args:
            path: Path to load the tasks from.

        Returns:
            Loaded tasks.

        """
        ...  # pragma: no cover

    def find_task_files(self, directory: TaskPath, pattern: str, limit: int = 0) -> list[TaskPath]:
        """Find task files in a directory.

        Args:
            directory: Directory to search in.
            pattern: Glob pattern to match files.
            limit: Maximum number of files to return.

        Returns:
            Sorted list of matching file paths.

        """
        ...  # pragma: no cover


class StageExecutor(Protocol):
    """Protocol for executing stages on task batches."""

    def execute_stage(
        self,
        stage: CuratorStage,
        task_batches: list[list[PipelineTask]],
        node_info: NodeInfo,
        worker_metadata: WorkerMetadata,
    ) -> list[list[PipelineTask]]:
        """Execute a stage on task batches.

        Args:
            stage: The stage to execute.
            task_batches: Batches of tasks to process.
            node_info: Node information for stage setup.
            worker_metadata: Worker metadata for stage setup.

        Returns:
            Processed task batches.

        """
        ...  # pragma: no cover


# ============================================================================
# Helper class for wrapping stages
# ============================================================================
class StageRunner:
    """Run a stage."""

    def __init__(self, stage: CuratorStage) -> None:
        """Initialize the stage runner.

        Args:
            stage: The stage to run.

        """
        self.stage = stage

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:
        """Set up the stage on the node.

        Args:
            node_info: The node info.
            worker_metadata: The worker metadata.

        """
        self.stage.setup_on_node(node_info, worker_metadata)

    def stage_setup(self) -> None:
        """Set up the stage."""
        self.stage.stage_setup()

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask] | None:
        """Process the data.

        Args:
            tasks: The tasks to process.

        Returns:
            Result of processing the tasks.

        """
        return self.stage.process_data(tasks)

    def destroy(self) -> None:
        """Destroy the stage runner."""
        self.stage.destroy()


# ============================================================================
# Default Implementations
# ============================================================================


class PickleTaskSerializer:
    """Default pickle-based task serializer."""

    def __init__(self, profile_name: str = "default") -> None:
        """Initialize the serializer."""
        self._profile_name = profile_name

    @contextmanager
    def _open_path(self, path: TaskPath, mode: str) -> Iterator[Any]:
        """Open a local or remote task path as a file-like object."""
        if isinstance(path, Path) and "w" in mode:
            storage_utils.create_path(str(path.parent))

        open_path: Path | str = path if isinstance(path, Path) else str(path)
        client = storage_utils.get_storage_client(str(path), profile_name=self._profile_name)
        client_params = storage_utils.get_smart_open_client_params(client) if client is not None else {}

        with smart_open.open(open_path, mode, **client_params) as f:
            yield f

    def save(self, path: TaskPath, tasks: list[PipelineTask]) -> None:
        """Save tasks to a pickle file."""
        with self._open_path(path, "wb") as f:
            pickle.dump(tasks, f)

    def load(self, path: TaskPath) -> list[PipelineTask]:
        """Load tasks from a pickle file."""
        logger.info(f"Loading tasks from {path}")
        with self._open_path(path, "rb") as f:
            return cast("list[PipelineTask]", pickle.load(f))  # noqa: S301

    def find_task_files(self, directory: TaskPath, pattern: str, limit: int = 0) -> list[TaskPath]:
        """Find task files matching a pattern."""
        client = storage_utils.get_storage_client(str(directory), profile_name=self._profile_name)
        relative_files = storage_utils.get_files_relative(str(directory), client, 0)
        matches = sorted(
            rel for rel in relative_files if Path(rel).parent == Path() and fnmatch.fnmatch(Path(rel).name, pattern)
        )
        files = [storage_utils.get_full_path(directory, match) for match in matches]
        if limit > 0:
            files = files[:limit]
        return files


class RayStageExecutor:
    """Ray-based stage executor for distributed processing."""

    def execute_stage(
        self,
        stage: CuratorStage,
        task_batches: list[list[PipelineTask]],
        node_info: NodeInfo,
        worker_metadata: WorkerMetadata,
    ) -> list[list[PipelineTask]]:
        """Execute a stage using Ray actors."""
        conda_env_name = stage.conda_env_name if stage.conda_env_name is not None else "default"
        runtime_env = PixiRuntimeEnv(conda_env_name)

        logger.info(f"Starting actor for stage {stage.__class__.__name__}")
        stage_runner: Any = (
            ray.remote(StageRunner)
            .options(
                runtime_env=runtime_env,
                num_cpus=stage.required_resources.cpus,
                num_gpus=stage.required_resources.gpus,
            )
            .remote(stage)
        )

        ray.get(stage_runner.setup_on_node.remote(node_info, worker_metadata))
        ray.get(stage_runner.stage_setup.remote())

        logger.info(f"Processing {len(task_batches)} task batches for stage {stage.__class__.__name__}")
        out_task_batches = []
        for task_batch in task_batches:
            result = ray.get(stage_runner.process_data.remote(task_batch))
            out_task_batch = result if result is not None else []
            out_task_batches.append(out_task_batch)
        logger.info(f"Processed {len(out_task_batches)} task batches for stage {stage.__class__.__name__}")

        ray.get(stage_runner.destroy.remote())
        ray.kill(stage_runner)

        return out_task_batches


class DirectStageExecutor:
    """Direct stage executor without Ray.

    Executes stages directly in the current process without using Ray actors.
    Useful for unit testing and debugging.

    """

    def execute_stage(
        self,
        stage: CuratorStage,
        task_batches: list[list[PipelineTask]],
        node_info: NodeInfo,
        worker_metadata: WorkerMetadata,
    ) -> list[list[PipelineTask]]:
        """Execute a stage directly without Ray."""
        logger.info(f"Executing stage {stage.__class__.__name__} directly (no Ray)")
        stage.setup_on_node(node_info, worker_metadata)
        stage.stage_setup()

        result = []
        for batch in task_batches:
            output = stage.process_data(batch)
            result.append(output if output is not None else [])

        stage.destroy()
        return result


def _clamp_sample_rate(value: float) -> float:
    """Clamp sample rate between 0.0 and 1.0."""
    return min(max(value, 0.0), 1.0)


@attrs.define
class StageSaveConfig:
    """Configuration for saving tasks from the pipeline.

    Args:
        path: Path to save tasks to.
        stages: List of stage names to save tasks from.
        sample_rate: Sample rate for saving tasks. Range is [0.0, 1.0].

    """

    path: TaskPath
    stages: list[str]
    sample_rate: float = attrs.field(converter=_clamp_sample_rate)
    profile_name: str = "default"


class TaskPathAllocator:
    """Allocate unique task-output paths without repeated directory listings."""

    _PATH_PATTERN = re.compile(r"^(?P<name>.+)_(?P<index>\d+)\.(?P<extension>.+)$")

    def __init__(self, base_output_path: TaskPath, *, profile_name: str = "default") -> None:
        """Seed allocation state from the existing output directory once."""
        self._base_output_path = base_output_path
        client = storage_utils.get_storage_client(str(base_output_path), profile_name=profile_name)
        existing_files = storage_utils.get_files_relative(str(base_output_path), client, 0)
        self._used_indices_by_key: dict[tuple[str, str], set[int]] = {}

        for relative_path in existing_files:
            match = self._PATH_PATTERN.fullmatch(relative_path)
            if match is None:
                continue
            key = (match.group("name"), match.group("extension"))
            self._used_indices_by_key.setdefault(key, set()).add(int(match.group("index")))

    def get_output_path(self, name: str, extension: str) -> TaskPath:
        """Return the next available output path for ``name`` and ``extension``."""
        key = (name, extension)
        used_indices = self._used_indices_by_key.setdefault(key, set())
        for i in range(1000):
            if i not in used_indices:
                used_indices.add(i)
                return storage_utils.get_full_path(self._base_output_path, f"{name}_{i:03d}.{extension}")
        msg = f"Failed to find a unique output path for {name}"
        raise RuntimeError(msg)


# ============================================================================
# Helper Functions
# ============================================================================


def _load_task_batches(
    path: TaskPath,
    limit: int,
    serializer: TaskSerializer | None = None,
    *,
    profile_name: str = "default",
) -> list[list[PipelineTask]]:
    """Load tasks from the tasks directory.

    Args:
        path: The path to the tasks directory.
        limit: The maximum number of task files to load.
        serializer: Task serializer to use. Defaults to PickleTaskSerializer.
        profile_name: Storage profile to use for remote paths.

    Returns:
        A list of task objects.

    """
    _serializer = serializer if serializer is not None else PickleTaskSerializer(profile_name=profile_name)
    files = _serializer.find_task_files(path, "*.pkl", limit)
    return [_serializer.load(file) for file in files]


def _get_task_suffix_from_input_video(input_video: object) -> str:
    """Return a basename for a supported video input path."""
    if isinstance(input_video, Path):
        return input_video.name

    if isinstance(input_video, storage_client.StoragePrefix):
        return Path(urlparse(str(input_video)).path).name

    if isinstance(input_video, str):
        if input_video.startswith(("s3://", "az://")):
            return Path(urlparse(input_video).path).name
        return Path(input_video).name

    msg = f"_get_task_suffix_from_input_video does not support input type: {type(input_video).__name__}"
    raise TypeError(msg)


def _get_name_from_tasks(class_name: str, tasks: list[PipelineTask]) -> str:
    """Get the name of the tasks.

    Args:
        class_name: The name of the class to save the tasks for.
        tasks: The tasks to get the name from.

    Returns:
        The name of the tasks.

    """
    if len(tasks) == 0:
        msg = "No tasks to get the name from"
        raise ValueError(msg)

    first_task = tasks[0]

    video = getattr(first_task, "video", None)
    if video is not None and isinstance(video, Video):
        task_suffix = _get_task_suffix_from_input_video(video.input_video)
    else:
        task_suffix = secrets.token_hex(8)

    return f"{class_name}/{task_suffix}"


def _save_tasks(
    class_name: str,
    config: StageSaveConfig,
    tasks: list[PipelineTask],
    serializer: TaskSerializer | None = None,
    allocator: TaskPathAllocator | None = None,
) -> None:
    """Save tasks to a pickle file.

    Args:
        class_name: The name of the class to save the tasks for.
        config: Configuration for saving stages for replay.
        tasks: The tasks to save.
        serializer: Task serializer to use. Defaults to PickleTaskSerializer.
        allocator: Output-path allocator. Defaults to TaskPathAllocator.

    """
    if serializer is None:
        serializer = PickleTaskSerializer(profile_name=config.profile_name)
    if allocator is None:
        allocator = TaskPathAllocator(config.path, profile_name=config.profile_name)

    name = _get_name_from_tasks(class_name, tasks)
    output_path = allocator.get_output_path(name, "task.pkl")
    serializer.save(output_path, tasks)
    logger.info(f"Saved tasks to {output_path}")


def _make_stage_save_class[T: CuratorStage](
    stage_cls: type[T],
    config: StageSaveConfig,
    serializer: TaskSerializer | None = None,
) -> type[T]:
    """Make a task-saving stage class.

    Create a new stage class that wraps the base stage class and save tasks
    passed to process_data to a file.

    Tasks are only saved if a random number between 0 and 1 is less than the
    sample rate.

    Args:
        stage_cls: The base stage class to wrap.
        config: Configuration for saving stages for replay.
        serializer: Task serializer. Defaults to PickleTaskSerializer.

    Returns:
        The task-saving stage class.

    """
    _serializer: TaskSerializer = (
        serializer if serializer is not None else PickleTaskSerializer(profile_name=config.profile_name)
    )

    if config.sample_rate <= 0.0:
        return stage_cls

    base_name = stage_cls.__name__

    class TaskSavingStage(stage_cls):  # type: ignore[valid-type, misc]
        _config = config
        _serializer_inst = _serializer
        _path_allocator = TaskPathAllocator(config.path, profile_name=config.profile_name)

        def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
            if tasks and random.random() <= self._config.sample_rate:  # noqa: S311
                _save_tasks(base_name, self._config, tasks, self._serializer_inst, self._path_allocator)

            return super().process_data(tasks)  # type: ignore[no-any-return]

    TaskSavingStage.__name__ = f"{base_name}WithTaskSaving"
    TaskSavingStage.__qualname__ = TaskSavingStage.__name__
    return TaskSavingStage


# ============================================================================
# Public API
# ============================================================================


def add_stage_replay_args(parser: argparse.ArgumentParser) -> None:
    """Add stage replay arguments to the parser.

    Args:
        parser: The parser to add the arguments to.

    """
    parser.add_argument(
        "--stage-save",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[],
        help="Comma-separated list of stage names to save input tasks from (e.g., 'Stage1,Stage2').",
    )
    parser.add_argument(
        "--stage-save-sample-rate",
        type=float,
        default=0.0,
        help="Fraction of tasks to save for each stage (0.0 = none, 1.0 = all)",
    )
    parser.add_argument(
        "--stage-replay",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[],
        help="Comma-separated list of stage names to replay using saved tasks. If one stage is provided, it will be "
        "in isolation. two stages are first_stage,last_stage. "
        "Saved tasks are loaded from --output-clip-path / tasks / stage_name / *.pkl",
    )
    parser.add_argument(
        "--stage-compare",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[],
        help=(
            "Comma-separated list of stage names to compare using saved tasks. "
            "If one stage is provided, compare that stage in isolation. "
            "If two stages are provided, compare the range start_stage,end_stage "
            "and use the final output as the golden comparison target."
        ),
    )
    parser.add_argument(
        "--stage-compare-path",
        type=str,
        default=None,
        help="Optional base path containing golden task pickles for --stage-compare. Defaults to --output-clip-path.",
    )
    parser.add_argument(
        "--stage-compare-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for numpy-based comparisons in --stage-compare. Default: 0.0",
    )
    parser.add_argument(
        "--stage-compare-pass-threshold",
        type=float,
        default=1.0,
        help="Minimum pass rate required for --stage-compare to exit successfully. Default: 1.0",
    )


def validate_stage_replay_args(args: argparse.Namespace) -> None:
    """Validate the stage replay arguments.

    Args:
        args: The arguments to validate.

    """
    stage_compare = getattr(args, "stage_compare", [])

    if len(args.stage_save) == 0 and len(args.stage_replay) == 0 and len(stage_compare) == 0:
        return

    enabled_modes = sum(int(len(value) > 0) for value in (args.stage_save, args.stage_replay, stage_compare))
    if enabled_modes > 1:
        msg = "Only one of --stage-save, --stage-replay, and --stage-compare may be used at a time"
        raise ValueError(msg)

    if len(args.stage_replay) > MAX_STAGE_REPLAY_ARGS:
        msg = "--stage-replay should only have one stage, or two stages: start, end."
        raise ValueError(msg)

    if len(stage_compare) > MAX_STAGE_COMPARE_ARGS:
        msg = "--stage-compare should only have one stage, or two stages: start, end."
        raise ValueError(msg)


def get_stages_to_replay(
    stages: list[CuratorStage | CuratorStageSpec], start_stage_name: str, end_stage_name: str
) -> list[CuratorStage]:
    """Get the replay stages from the stages list.

    Args:
        stages: The list of stages.
        start_stage_name: The name of the start stage.
        end_stage_name: The name of the end stage.

    Returns:
        A list of stages to replay.

    Raises:
        ValueError: If the end stages precedes the start stage in the pipeline, or if
            either stage is not found.

    """
    replay_stages: list[CuratorStage] = []
    started = False
    found_end = False

    for stage in stages:
        _stage = cast("CuratorStage", stage.stage) if isinstance(stage, CuratorStageSpec) else stage
        name = _stage.__class__.__name__

        # Check if this is the start stage
        if name == start_stage_name:
            started = True

        # Add stage if we've started collecting
        if started:
            replay_stages.append(_stage)

        # Check if this is the end stage
        if name == end_stage_name:
            if not started:
                msg = f"Stage {end_stage_name} is the first stage found, but it should be the last stage"
                raise ValueError(msg)
            found_end = True
            break

    if len(replay_stages) == 0:
        msg = f"No stages found to replay, stages {start_stage_name}, {end_stage_name} not present"
        raise ValueError(msg)

    if not found_end:
        msg = f"End stage {end_stage_name} not found in pipeline"
        raise ValueError(msg)

    return replay_stages


def run_stage_replay(  # noqa: PLR0913
    stages: list[CuratorStage],
    path: TaskPath,
    limit: int,
    *,
    profile_name: str = "default",
    executor: StageExecutor | None = None,
    serializer: TaskSerializer | None = None,
    init_ray: bool = True,
) -> list[PipelineTask]:
    """Replay stages with task pickles.

    Args:
        stages: list of stages to replay.
        path: The path to the tasks directory that holds the task pickles.
        limit: The maximum number of tasks to load.
        profile_name: Storage profile to use for remote paths.
        executor: Stage executor to use. Defaults to RayStageExecutor.
        serializer: Task serializer to use. Defaults to PickleTaskSerializer.
        init_ray: Whether to initialize Ray. Set to False if Ray is already initialized.

    Returns:
        A list of task objects.

    """
    if len(stages) == 0:
        msg = "No stages to replay"
        raise ValueError(msg)

    _executor = executor if executor is not None else RayStageExecutor()
    _serializer = serializer if serializer is not None else PickleTaskSerializer(profile_name=profile_name)

    start_stage = stages[0]
    end_stage = stages[-1]
    logger.info(
        f"Running isolated stages {start_stage.__class__.__name__} -> {end_stage.__class__.__name__} "
        f"loading input tasks from {path}, {limit=}"
    )

    node_info, worker_metadata = NodeInfo(node_id="localhost"), WorkerMetadata.make_dummy()

    if init_ray and not ray.is_initialized():
        ray.init()

    task_batches = _load_task_batches(path, limit, _serializer, profile_name=profile_name)

    if len(task_batches) == 0:
        msg = f"No input tasks found in {path}"
        raise ValueError(msg)

    for stage in stages:
        task_batches = _executor.execute_stage(stage, task_batches, node_info, worker_metadata)

    return [task for batch in task_batches for task in batch]


def should_save_stage(stage: CuratorStage | CuratorStageSpec, config: StageSaveConfig) -> bool:
    """Check if the stage should be saved.

    Args:
        stage: The stage to check.
        config: Configuration for saving stages for replay.

    Returns:
        True if the stage should be saved, False otherwise.

    """
    _stage = cast("CuratorStage", stage.stage) if isinstance(stage, CuratorStageSpec) else stage
    return _stage.__class__.__name__ in config.stages


def stage_save_wrapper(
    stage: CuratorStage | CuratorStageSpec,
    config: StageSaveConfig,
) -> CuratorStage | CuratorStageSpec:
    """Wrap the process_data method of a stage so that it saves tasks.

    This function modifies the stage's class in place, so that the stage's
    state is preserved.

    The new class is a subclass of the stage's original class, and that class
    overrides process_data method to save tasks.

    Args:
        stage: The stage to wrap.
        config: Configuration for saving stages for replay.

    Returns:
        The stage or stage spec with the process_data method wrapped.

    """
    _stage = cast("CuratorStage", stage.stage) if isinstance(stage, CuratorStageSpec) else stage
    name = _stage.__class__.__name__

    logger.info(f"Wrapping process_data for stage {name} with path {config.path} and sample_rate {config.sample_rate}")

    # Swap the instance's class in place, keeping all attributes as-is.
    _stage.__class__ = _make_stage_save_class(_stage.__class__, config)

    if isinstance(stage, CuratorStage):
        return _stage

    stage.stage = _stage
    return stage
