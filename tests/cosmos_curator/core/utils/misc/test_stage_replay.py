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
"""Tests for cosmos_curator.core.utils.misc.stage_replay."""

import argparse
import io
import pickle
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import attrs
import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec, PipelineTask
from cosmos_curator.core.utils.misc.stage_replay import (
    DirectStageExecutor,
    PickleTaskSerializer,
    RayStageExecutor,
    StageRunner,
    StageSaveConfig,
    TaskPathAllocator,
    _get_name_from_tasks,
    _load_task_batches,
    _make_stage_save_class,
    _save_tasks,
    add_stage_replay_args,
    get_stages_to_replay,
    run_stage_replay,
    should_save_stage,
    stage_save_wrapper,
    validate_stage_replay_args,
)
from cosmos_curator.core.utils.storage.azure_client import AzurePrefix
from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.pipelines.video.utils.data_model import Video
from cosmos_xenna.pipelines.private.resources import NodeInfo, Resources, WorkerMetadata


@attrs.define
class TestTask(PipelineTask):
    """Test task for unit testing."""

    value: int = 0


@attrs.define
class VideoTask(PipelineTask):
    """Test task with video attribute for unit testing."""

    video: Video | None = None


class TestStage(CuratorStage):
    """Test stage for unit testing."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Process data by returning the tasks unchanged."""
        return tasks


@pytest.mark.parametrize(
    ("tasks", "expected_result_type", "expected_suffix", "raises"),
    [
        # Empty list should raise ValueError
        ([], "error", "", pytest.raises(ValueError, match=r".*")),
        # Task with Video and Path input_video should use video name
        (
            [VideoTask(video=Video(input_video=Path("/path/to/video.mp4")))],
            "exact",
            "TestClassName/video.mp4",
            nullcontext(),
        ),
        # Task with Video and local string input_video should use basename
        (
            [VideoTask(video=Video(input_video="/path/to/video.mp4"))],
            "exact",
            "TestClassName/video.mp4",
            nullcontext(),
        ),
        # Task with Video and S3 input_video should use object basename
        (
            [VideoTask(video=Video(input_video="s3://bucket/path/to/video.mp4"))],
            "exact",
            "TestClassName/video.mp4",
            nullcontext(),
        ),
        # Task with Video and S3 StoragePrefix should use object basename
        (
            [VideoTask(video=Video(input_video=S3Prefix("s3://bucket/path/to/video.mp4")))],
            "exact",
            "TestClassName/video.mp4",
            nullcontext(),
        ),
        # Task with Video and Azure StoragePrefix should use object basename
        (
            [VideoTask(video=Video(input_video=AzurePrefix("az://container/path/to/video.mp4")))],
            "exact",
            "TestClassName/video.mp4",
            nullcontext(),
        ),
        # Task with Video and bare filename string should use basename directly
        (
            [VideoTask(video=Video(input_video="string_path"))],
            "exact",
            "TestClassName/string_path",
            nullcontext(),
        ),
        # Task without video attribute should use random hex
        ([TestTask(value=1)], "random_hex", "", nullcontext()),
        # Task with non-Video video attribute should use random hex
        ([VideoTask(video="not_a_video")], "random_hex", "", nullcontext()),
        # Task with None video should use random hex
        ([VideoTask(video=None)], "random_hex", "", nullcontext()),
    ],
)
def test_get_name_from_tasks(
    tasks: list[PipelineTask],
    expected_result_type: str,
    expected_suffix: str,
    raises: AbstractContextManager[Any],
) -> None:
    """Test _get_name_from_tasks with various task configurations."""
    class_name = "TestClassName"

    with raises:
        result = _get_name_from_tasks(class_name, tasks)
        if expected_result_type == "exact":
            assert result == expected_suffix
        elif expected_result_type == "random_hex":
            # Should be in format "TestClassName/{16_char_hex}"
            assert result.startswith(f"{class_name}/")
            suffix = result.split("/", 1)[1]
            # secrets.token_hex(8) produces 16 hex characters
            assert len(suffix) == 16
            assert all(c in "0123456789abcdef" for c in suffix)


@pytest.mark.parametrize(
    ("input_rate", "expected_rate"),
    [
        (-1.0, 0.0),  # Below minimum - clamp to 0.0
        (-0.5, 0.0),  # Below minimum - clamp to 0.0
        (0.0, 0.0),  # At minimum - no change
        (0.5, 0.5),  # Within range - no change
        (1.0, 1.0),  # At maximum - no change
        (1.5, 1.0),  # Above maximum - clamp to 1.0
        (2.0, 1.0),  # Above maximum - clamp to 1.0
        (100.0, 1.0),  # Far above maximum - clamp to 1.0
    ],
)
def test_stage_save_config_sample_rate_clamping(tmp_path: Path, input_rate: float, expected_rate: float) -> None:
    """Test that StageSaveConfig clamps sample_rate to [0.0, 1.0] range."""
    config = StageSaveConfig(path=tmp_path, stages=["TestStage"], sample_rate=input_rate)
    assert config.sample_rate == expected_rate


@pytest.mark.parametrize(
    ("stage", "config_stages", "expected_result"),
    [
        # CuratorStage with name in config - should return True
        (TestStage(), ["TestStage"], True),
        # CuratorStage with name NOT in config - should return False
        (TestStage(), ["SomeOtherStage"], False),
        # CuratorStage with empty stages list - should return False
        (TestStage(), [], False),
        # CuratorStage with multiple stages in config, matching - should return True
        (TestStage(), ["OtherStage1", "TestStage", "OtherStage2"], True),
        # CuratorStage with multiple stages in config, not matching - should return False
        (TestStage(), ["OtherStage1", "OtherStage2", "OtherStage3"], False),
        # CuratorStageSpec with stage name in config - should return True
        (CuratorStageSpec(TestStage()), ["TestStage"], True),
        # CuratorStageSpec with stage name NOT in config - should return False
        (CuratorStageSpec(TestStage()), ["DifferentStage"], False),
        # CuratorStageSpec with empty stages list - should return False
        (CuratorStageSpec(TestStage()), [], False),
        # CuratorStageSpec with multiple stages, matching - should return True
        (CuratorStageSpec(TestStage()), ["OtherStage", "TestStage", "AnotherStage"], True),
    ],
)
def test_should_save_stage(
    tmp_path: Path, stage: CuratorStage | CuratorStageSpec, config_stages: list[str], *, expected_result: bool
) -> None:
    """Test should_save_stage with various stage types and configurations."""
    config = StageSaveConfig(path=tmp_path, stages=config_stages, sample_rate=1.0)
    result = should_save_stage(stage, config)
    assert result == expected_result


@pytest.mark.parametrize(
    ("sample_rate", "stage_in_list", "expected_files", "verify_contents", "use_spec"),
    [
        (1.0, True, 1, True, False),  # Save with rate 1.0
        (0.0, True, 0, False, False),  # Don't save with rate 0.0
        (0.0, False, 0, False, False),  # Not in list - no wrapping
        (1.0, True, 1, True, True),  # Save with CuratorStageSpec wrapper
    ],
)
def test_stage_save_wrapper(  # noqa: PLR0913
    tmp_path: Path,
    sample_rate: float,
    expected_files: int,
    *,
    stage_in_list: bool,
    verify_contents: bool,
    use_spec: bool,
) -> None:
    """Test stage_save_wrapper with various configurations."""
    test_stage = TestStage()
    test_tasks = [TestTask(value=1), TestTask(value=2), TestTask(value=3)]
    output_path = tmp_path / "debug_output"

    stages_list = [TestStage.__name__] if stage_in_list else ["SomeOtherStage"]
    config = StageSaveConfig(path=output_path, stages=stages_list, sample_rate=sample_rate)

    # Wrap in CuratorStageSpec if requested
    stage_input: CuratorStage | CuratorStageSpec = CuratorStageSpec(test_stage) if use_spec else test_stage
    original_class = test_stage.__class__

    wrapped_stage = stage_save_wrapper(stage_input, config)

    # Extract the actual stage for processing
    _wrapped_stage = (
        cast("CuratorStage", wrapped_stage.stage) if isinstance(wrapped_stage, CuratorStageSpec) else wrapped_stage
    )
    result = _wrapped_stage.process_data(test_tasks)  # type: ignore[arg-type]

    assert result == test_tasks

    # Check if class was wrapped
    if not stage_in_list:
        if use_spec:
            assert isinstance(wrapped_stage, CuratorStageSpec)
            assert wrapped_stage.stage.__class__ is original_class
        else:
            assert wrapped_stage.__class__ is original_class

    # Verify return type matches input type
    if use_spec:
        assert isinstance(wrapped_stage, CuratorStageSpec)
    else:
        assert isinstance(wrapped_stage, CuratorStage)

    # Verify file count
    pickle_files = list(output_path.glob("**/*.task.pkl"))
    assert len(pickle_files) == expected_files

    # Verify file contents if expected
    if verify_contents and expected_files > 0:
        with pickle_files[0].open("rb") as f:
            loaded_tasks = pickle.load(f)  # noqa: S301

        assert len(loaded_tasks) == len(test_tasks)
        for loaded, original in zip(loaded_tasks, test_tasks, strict=True):
            assert isinstance(loaded, TestTask)
            assert loaded.value == original.value


# ============================================================================
# Tests for get_stages_to_replay
# ============================================================================


class Stage1(CuratorStage):
    """Test stage 1."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Process data."""
        return tasks


class Stage2(CuratorStage):
    """Test stage 2."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Process data."""
        return tasks


class Stage3(CuratorStage):
    """Test stage 3."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Process data."""
        return tasks


class Stage4(CuratorStage):
    """Test stage 4."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Process data."""
        return tasks


@pytest.mark.parametrize(
    ("stages", "start_name", "end_name", "expected_names", "raises"),
    [
        # Single stage
        ([Stage1(), Stage2(), Stage3()], "Stage2", "Stage2", ["Stage2"], nullcontext()),
        # Range of stages
        ([Stage1(), Stage2(), Stage3(), Stage4()], "Stage2", "Stage3", ["Stage2", "Stage3"], nullcontext()),
        # All stages
        ([Stage1(), Stage2(), Stage3()], "Stage1", "Stage3", ["Stage1", "Stage2", "Stage3"], nullcontext()),
        # With CuratorStageSpec
        (
            [CuratorStageSpec(Stage1()), CuratorStageSpec(Stage2()), CuratorStageSpec(Stage3())],
            "Stage1",
            "Stage2",
            ["Stage1", "Stage2"],
            nullcontext(),
        ),
        # Mixed stages and specs
        ([Stage1(), CuratorStageSpec(Stage2()), Stage3()], "Stage2", "Stage3", ["Stage2", "Stage3"], nullcontext()),
        # End before start - error
        ([Stage1(), Stage2(), Stage3()], "Stage3", "Stage1", [], pytest.raises(ValueError, match=r".*")),
        # Stages not found - error
        (
            [Stage1(), Stage2(), Stage3()],
            "NonExistent",
            "AlsoNonExistent",
            [],
            pytest.raises(ValueError, match=r".*"),
        ),
        # Start exists but end stage missing - error
        (
            [Stage1(), Stage2(), Stage3()],
            "Stage1",
            "Stage4",
            [],
            pytest.raises(ValueError, match=r"End stage Stage4 not found in pipeline"),
        ),
    ],
)
def test_get_stages_to_replay(
    stages: list[CuratorStage | CuratorStageSpec],
    start_name: str,
    end_name: str,
    expected_names: list[str],
    raises: AbstractContextManager[Any],
) -> None:
    """Test get_stages_to_replay with various stage configurations."""
    with raises:
        result = get_stages_to_replay(stages, start_name, end_name)
        assert [s.__class__.__name__ for s in result] == expected_names


# ============================================================================
# Tests for argparse functions
# ============================================================================


@pytest.mark.parametrize(
    ("stage_save", "stage_replay", "stage_compare", "raises"),
    [
        ([], [], [], nullcontext()),  # No args
        (["Stage1", "Stage2"], [], [], nullcontext()),  # Save only
        ([], ["Stage1"], [], nullcontext()),  # Replay one stage
        ([], ["Stage1", "Stage2"], [], nullcontext()),  # Replay two stages
        ([], [], ["Stage1"], nullcontext()),  # Compare one stage
        ([], [], ["Stage1", "Stage2"], nullcontext()),  # Compare two stages
        (
            ["Stage1"],
            ["Stage1"],
            [],
            pytest.raises(ValueError, match="Only one of --stage-save, --stage-replay, and --stage-compare"),
        ),  # Save + replay
        (
            ["Stage1"],
            [],
            ["Stage1"],
            pytest.raises(ValueError, match="Only one of --stage-save, --stage-replay, and --stage-compare"),
        ),  # Save + compare
        (
            [],
            ["Stage1"],
            ["Stage1"],
            pytest.raises(ValueError, match="Only one of --stage-save, --stage-replay, and --stage-compare"),
        ),  # Replay + compare
        ([], ["Stage1", "Stage2", "Stage3"], [], pytest.raises(ValueError, match="should only have one stage, or two")),
        ([], [], ["Stage1", "Stage2", "Stage3"], pytest.raises(ValueError, match="should only have one stage, or two")),
    ],
)
def test_validate_stage_replay_args(
    stage_save: list[str],
    stage_replay: list[str],
    stage_compare: list[str],
    raises: AbstractContextManager[Any],
) -> None:
    """Test validation of stage replay arguments."""
    args = argparse.Namespace(
        stage_save=stage_save,
        stage_replay=stage_replay,
        stage_compare=stage_compare,
    )
    with raises:
        validate_stage_replay_args(args)


def test_add_stage_replay_args_smoke() -> None:
    """Smoke test for add_stage_replay_args - ensure it runs without errors.

    This test intentionally skips checking for the added arguments to avoid
    relying on specific argument names that may change.
    """
    parser = argparse.ArgumentParser()
    add_stage_replay_args(parser)


# ============================================================================
# Tests for _load_task_batches
# ============================================================================


class MockTaskSerializer:
    """Mock task serializer for testing."""

    def __init__(self, files: list[Path], task_data: dict[Path, list[PipelineTask]]) -> None:
        """Initialize mock serializer.

        Args:
            files: List of file paths to return from find_task_files.
            task_data: Mapping of file paths to task lists.

        """
        self.files = files
        self.task_data = task_data
        self.loaded_files: list[Path] = []

    def find_task_files(self, directory: Path, pattern: str, limit: int = 0) -> list[Path]:  # noqa: ARG002
        """Return the predefined list of files."""
        if limit > 0:
            return self.files[:limit]
        return self.files

    def load(self, path: Path) -> list[PipelineTask]:
        """Load tasks from the mock data."""
        self.loaded_files.append(path)
        return self.task_data.get(path, [])

    def save(self, path: Path, tasks: list[PipelineTask]) -> None:
        """Save tasks (not used in these tests)."""
        # Intentionally empty for mock


@pytest.mark.parametrize(
    ("limit", "expected_count", "expected_values"),
    [
        (0, 3, [1, 2, 3]),  # No limit - load all
        (2, 2, [1, 2]),  # With limit - load only first 2
        (5, 3, [1, 2, 3]),  # Limit larger than available files - load all
        (1, 1, [1]),  # Limit of 1 - load only first
    ],
)
def test_load_task_batches_with_limit(
    tmp_path: Path, limit: int, expected_count: int, expected_values: list[int]
) -> None:
    """Test loading task batches respects limit parameter."""
    file1 = tmp_path / "task_000.pkl"
    file2 = tmp_path / "task_001.pkl"
    file3 = tmp_path / "task_002.pkl"

    task_data = {
        file1: [TestTask(value=1)],
        file2: [TestTask(value=2)],
        file3: [TestTask(value=3)],
    }
    serializer = MockTaskSerializer([file1, file2, file3], task_data)  # type: ignore[arg-type]

    result = _load_task_batches(tmp_path, limit=limit, serializer=serializer)

    assert len(result) == expected_count
    for i, expected_value in enumerate(expected_values):
        assert len(result[i]) == 1
        assert result[i][0].value == expected_value  # type: ignore[attr-defined]


def test_load_task_batches_empty_directory(tmp_path: Path) -> None:
    """Test loading task batches from empty directory."""
    serializer = MockTaskSerializer([], {})
    result = _load_task_batches(tmp_path, limit=0, serializer=serializer)
    assert len(result) == 0


def test_load_task_batches_uses_pickle_by_default(tmp_path: Path) -> None:
    """Test that _load_task_batches uses PickleTaskSerializer by default."""
    # Create a real pickle file
    task_file = tmp_path / "task_000.pkl"
    tasks = [TestTask(value=42)]
    serializer = PickleTaskSerializer()
    serializer.save(task_file, tasks)  # type: ignore[arg-type]

    # Load without providing serializer (should use default)
    result = _load_task_batches(tmp_path, limit=0)

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0].value == 42  # type: ignore[attr-defined]


def test_pickle_task_serializer_find_task_files_sorts_and_limits(tmp_path: Path) -> None:
    """Test that PickleTaskSerializer returns sorted task files and respects limit."""
    serializer = PickleTaskSerializer()
    filenames = ["task_010.pkl", "task_002.pkl", "task_001.pkl"]
    for name in filenames:
        (tmp_path / name).write_bytes(b"")

    result = serializer.find_task_files(tmp_path, "*.pkl", limit=2)

    assert result == [tmp_path / "task_001.pkl", tmp_path / "task_002.pkl"]


@pytest.mark.parametrize(
    "case",
    [
        ("remote", "test-profile", {"transport_params": {"client": "mock-client"}}, None, 7),
        ("local", "default", {}, "nested", 13),
    ],
)
def test_pickle_task_serializer_save_uses_smart_open(
    case: tuple[str, str, dict[str, Any], str | None, int],
    tmp_path: Path,
) -> None:
    """Test save uses smart_open for both local and remote task paths."""
    path_kind, profile_name, expected_kwargs, expect_create_subdir, task_value = case
    serializer = PickleTaskSerializer(profile_name=profile_name)
    tasks = [TestTask(value=task_value)]
    buffer = io.BytesIO()
    client = MagicMock()
    if path_kind == "remote":
        path = S3Prefix("s3://bucket/tasks/task_000.pkl")
        expected_open_path: str | Path = "s3://bucket/tasks/task_000.pkl"
    else:
        path = tmp_path / "nested" / "task_000.pkl"
        expected_open_path = path

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_storage_client", return_value=client),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_smart_open_client_params",
            return_value=expected_kwargs,
        ),
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.create_path") as mock_create_path,
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.smart_open.open",
            return_value=nullcontext(buffer),
        ) as mock_open,
    ):
        serializer.save(path, tasks)

    loaded = cast("list[TestTask]", pickle.loads(buffer.getvalue()))  # noqa: S301
    assert loaded[0].value == task_value
    if expect_create_subdir is None:
        mock_create_path.assert_not_called()
    else:
        mock_create_path.assert_called_once_with(str(tmp_path / expect_create_subdir))
    mock_open.assert_called_once_with(expected_open_path, "wb", **expected_kwargs)


@pytest.mark.parametrize(
    "case",
    [
        ("remote", "test-profile", {"transport_params": {"client": "mock-client"}}, 11),
        ("local", "default", {}, 17),
    ],
)
def test_pickle_task_serializer_load_uses_smart_open(
    case: tuple[str, str, dict[str, Any], int],
    tmp_path: Path,
) -> None:
    """Test load uses smart_open for both local and remote task paths."""
    path_kind, profile_name, expected_kwargs, task_value = case
    serializer = PickleTaskSerializer(profile_name=profile_name)
    buffer = io.BytesIO()
    pickle.dump([TestTask(value=task_value)], buffer)
    buffer.seek(0)
    client = MagicMock()
    if path_kind == "remote":
        path = S3Prefix("s3://bucket/tasks/task_000.pkl")
        expected_open_path: str | Path = "s3://bucket/tasks/task_000.pkl"
    else:
        path = tmp_path / "task_000.pkl"
        expected_open_path = path

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_storage_client", return_value=client),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_smart_open_client_params",
            return_value=expected_kwargs,
        ),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.smart_open.open",
            return_value=nullcontext(buffer),
        ) as mock_open,
    ):
        loaded = serializer.load(path)

    assert cast("TestTask", loaded[0]).value == task_value
    mock_open.assert_called_once_with(expected_open_path, "rb", **expected_kwargs)


def test_pickle_task_serializer_find_task_files_remote_sorts_and_limits() -> None:
    """Test that remote task discovery returns sorted fully-qualified paths."""
    serializer = PickleTaskSerializer(profile_name="test-profile")
    client = MagicMock()

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_storage_client", return_value=client),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_files_relative",
            return_value=[
                "task_010.pkl",
                "task_002.pkl",
                "task_001.pkl",
            ],
        ) as mock_get_files,
    ):
        result = serializer.find_task_files(S3Prefix("s3://bucket/tasks"), "*.pkl", limit=2)

    assert result == [
        S3Prefix("s3://bucket/tasks/task_001.pkl"),
        S3Prefix("s3://bucket/tasks/task_002.pkl"),
    ]
    mock_get_files.assert_called_once_with("s3://bucket/tasks", client, 0)


def test_pickle_task_serializer_find_task_files_ignores_nested_relative_paths() -> None:
    """Test that serializer discovery only matches top-level files for a glob like ``*.pkl``."""
    serializer = PickleTaskSerializer(profile_name="test-profile")
    client = MagicMock()

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_storage_client", return_value=client),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_files_relative",
            return_value=[
                "subdir/task_000.pkl",
                "task_002.pkl",
                "task_001.pkl",
            ],
        ),
    ):
        result = serializer.find_task_files(S3Prefix("s3://bucket/tasks"), "*.pkl")

    assert result == [
        S3Prefix("s3://bucket/tasks/task_001.pkl"),
        S3Prefix("s3://bucket/tasks/task_002.pkl"),
    ]


def test_task_path_allocator_uses_single_listing_for_repeated_allocations() -> None:
    """Test TaskPathAllocator lists existing files once and allocates gaps locally."""
    client = MagicMock()

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_storage_client", return_value=client),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_files_relative",
            return_value=[
                "TestStage/test_file_000.task.pkl",
                "TestStage/test_file_001.task.pkl",
                "TestStage/test_file_005.task.pkl",
            ],
        ) as mock_get_files_relative,
    ):
        allocator = TaskPathAllocator(S3Prefix("s3://bucket/tasks"), profile_name="test-profile")
        first = allocator.get_output_path("TestStage/test_file", "task.pkl")
        second = allocator.get_output_path("TestStage/test_file", "task.pkl")

    assert first == S3Prefix("s3://bucket/tasks/TestStage/test_file_002.task.pkl")
    assert second == S3Prefix("s3://bucket/tasks/TestStage/test_file_003.task.pkl")
    mock_get_files_relative.assert_called_once_with("s3://bucket/tasks", client, 0)


def test_load_task_batches_remote_accepts_storage_prefix() -> None:
    """Test that _load_task_batches uses StoragePrefix task paths for remote loading."""
    serializer = MagicMock()
    task_path = S3Prefix("s3://bucket/tasks")
    serializer.find_task_files.return_value = [
        S3Prefix("s3://bucket/tasks/task_001.pkl"),
        S3Prefix("s3://bucket/tasks/task_002.pkl"),
    ]
    serializer.load.side_effect = [[TestTask(value=1)], [TestTask(value=2)]]

    result = _load_task_batches(task_path, limit=2, serializer=serializer)

    assert [cast("TestTask", batch[0]).value for batch in result] == [1, 2]
    serializer.find_task_files.assert_called_once_with(task_path, "*.pkl", 2)


# ============================================================================
# Tests for deterministic sampling
# ============================================================================


class SaveCountingStage(CuratorStage):
    """Test stage that counts how many times tasks are saved."""

    save_count = 0

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Process data and return tasks unchanged."""
        return tasks


@pytest.mark.parametrize(
    ("random_value", "sample_rate", "tasks", "num_times", "expected_saved_count"),
    [
        (0.0, 0.5, [TestTask(value=1)], 3, 3),  # Always save (0.0 <= 0.5)
        (1.0, 0.5, [TestTask(value=1)], 3, 0),  # Never save (1.0 > 0.5)
        (0.5, 0.5, [TestTask(value=1)], 1, 1),  # Boundary (0.5 <= 0.5)
        (0.0, 1.0, [], 1, 0),  # Empty tasks never saved
    ],
)
def test_make_stage_save_class_deterministic(  # noqa: PLR0913
    tmp_path: Path,
    random_value: float,
    sample_rate: float,
    tasks: list[PipelineTask],
    num_times: int,
    expected_saved_count: int,
) -> None:
    """Test _make_stage_save_class with deterministic random sampling."""
    config = StageSaveConfig(
        path=tmp_path,
        stages=["SaveCountingStage"],
        sample_rate=sample_rate,
    )

    with patch("random.random", return_value=random_value):
        stage_cls = _make_stage_save_class(
            SaveCountingStage,
            config,
        )

        stage = stage_cls()
        for _ in range(num_times):
            stage.process_data(tasks)

    saved_files = list(tmp_path.glob("**/*.pkl"))
    assert len(saved_files) == expected_saved_count


def test_make_stage_save_class_uses_config_profile_name() -> None:
    """Test _make_stage_save_class propagates config.profile_name to the default serializer."""
    config = StageSaveConfig(
        path=S3Prefix("s3://bucket/tasks"),
        stages=["SaveCountingStage"],
        sample_rate=1.0,
        profile_name="test-profile",
    )

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_storage_client", return_value=MagicMock()),
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_files_relative", return_value=[]),
    ):
        stage_cls = _make_stage_save_class(SaveCountingStage, config)

    assert stage_cls._serializer_inst._profile_name == "test-profile"  # type: ignore[attr-defined]


def test_make_stage_save_class_uses_single_listing_for_repeated_saves() -> None:
    """Test wrapped save stages do not relist the output directory for each saved batch."""
    config = StageSaveConfig(
        path=S3Prefix("s3://bucket/tasks"),
        stages=["SaveCountingStage"],
        sample_rate=1.0,
        profile_name="test-profile",
    )
    tasks = [VideoTask(video=Video(input_video=S3Prefix("s3://bucket/path/to/video.mp4")))]

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.random.random", return_value=0.0),
        patch("cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_storage_client", return_value=MagicMock()),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_smart_open_client_params", return_value={}
        ),
        patch(
            "cosmos_curator.core.utils.misc.stage_replay.storage_utils.get_files_relative",
            return_value=[],
        ) as mock_get_files_relative,
        patch("cosmos_curator.core.utils.misc.stage_replay.smart_open.open", return_value=nullcontext(io.BytesIO())),
    ):
        stage_cls = _make_stage_save_class(SaveCountingStage, config)
        stage = stage_cls()
        stage.process_data(tasks)
        stage.process_data(tasks)

    mock_get_files_relative.assert_called_once()


def test_save_tasks_default_serializer(tmp_path: Path) -> None:
    """Test _save_tasks uses default PickleTaskSerializer when serializer is None."""
    output_path = tmp_path / "debug_output"
    config = StageSaveConfig(path=output_path, stages=["TestStage"], sample_rate=1.0)
    tasks = [TestTask(value=1), TestTask(value=2)]

    # Call _save_tasks without providing a serializer
    _save_tasks("TestStage", config, tasks, serializer=None)  # type: ignore[arg-type]

    # Verify task was saved using default PickleTaskSerializer
    pickle_files = list(output_path.glob("**/*.pkl"))
    assert len(pickle_files) == 1

    # Verify contents can be loaded
    with pickle_files[0].open("rb") as f:
        loaded_tasks = pickle.load(f)  # noqa: S301

    assert len(loaded_tasks) == len(tasks)
    for loaded, original in zip(loaded_tasks, tasks, strict=True):
        assert isinstance(loaded, TestTask)
        assert loaded.value == original.value


# ============================================================================
# Tests for run_stage_replay with DirectStageExecutor
# ============================================================================


class IncrementStage(CuratorStage):
    """Test stage that increments task values."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Increment value of each task."""
        return [TestTask(value=task.value + 1) if isinstance(task, TestTask) else task for task in tasks]


class DoubleStage(CuratorStage):
    """Test stage that doubles task values."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Double value of each task."""
        return [TestTask(value=task.value * 2) if isinstance(task, TestTask) else task for task in tasks]


class FilterStage(CuratorStage):
    """Test stage that filters out even values."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Filter out tasks with even values."""
        return [task for task in tasks if isinstance(task, TestTask) and task.value % 2 == 1]


def test_run_stage_replay_no_stages() -> None:
    """Test replay with no stages."""
    with pytest.raises(ValueError, match=r".*"):
        run_stage_replay(
            stages=[],
            path=Path("/nope"),
            limit=0,
            executor=DirectStageExecutor(),
            serializer=PickleTaskSerializer(),
            init_ray=False,
        )


def test_run_stage_replay_no_input_tasks(tmp_path: Path) -> None:
    """Test replay with empty task directory (no input tasks)."""
    # Create an empty directory with no task files
    empty_dir = tmp_path / "empty_tasks"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match=r"No input tasks found in .*"):
        run_stage_replay(
            stages=[IncrementStage()],
            path=empty_dir,
            limit=0,
            executor=DirectStageExecutor(),
            serializer=PickleTaskSerializer(),
            init_ray=False,
        )


@pytest.mark.parametrize(
    ("task_batches", "stages", "limit", "expected_values"),
    [
        # Single stage, single batch with multiple tasks
        ([[TestTask(value=5), TestTask(value=10)]], [IncrementStage()], 1, [6, 11]),
        # Multiple stages in sequence
        ([[TestTask(value=5)]], [IncrementStage(), DoubleStage()], 1, [12]),
        # Single stage, multiple batches
        (
            [[TestTask(value=1)], [TestTask(value=2)], [TestTask(value=3)]],
            [DoubleStage()],
            0,
            [2, 4, 6],
        ),
    ],
)
def test_run_stage_replay(
    tmp_path: Path,
    task_batches: list[list[TestTask]],
    stages: list[CuratorStage],
    limit: int,
    expected_values: list[int],
) -> None:
    """Test stage replay with DirectStageExecutor."""
    serializer = PickleTaskSerializer()
    for i, batch in enumerate(task_batches):
        serializer.save(tmp_path / f"batch_{i:03d}.pkl", batch)  # type: ignore[arg-type]

    executor = DirectStageExecutor()
    result = run_stage_replay(
        stages=stages,
        path=tmp_path,
        limit=limit,
        executor=executor,
        serializer=serializer,
        init_ray=False,
    )

    assert len(result) == len(expected_values)
    for task, expected_value in zip(result, expected_values, strict=True):
        assert task.value == expected_value  # type: ignore[attr-defined]


@pytest.mark.parametrize("init_ray", [True, False])
def test_run_stage_replay_init_ray(tmp_path: Path, *, init_ray: bool) -> None:
    """Test that ray.init() is called based on init_ray parameter."""
    # Setup minimal test data
    serializer = PickleTaskSerializer()
    serializer.save(tmp_path / "batch_000.pkl", [TestTask(value=1)])

    # Mock ray.init and ray.is_initialized to avoid actually initializing Ray
    # and to ensure the init_ray guard doesn't short-circuit when Ray is
    # already initialized from a prior test in the same process.
    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.init") as mock_ray_init,
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.is_initialized", return_value=False),
    ):
        result = run_stage_replay(
            stages=[IncrementStage()],
            path=tmp_path,
            limit=1,
            executor=DirectStageExecutor(),
            serializer=serializer,
            init_ray=init_ray,
        )

        # Verify ray.init was called based on init_ray parameter
        if init_ray:
            mock_ray_init.assert_called_once()
        else:
            mock_ray_init.assert_not_called()

        # Verify the function still works correctly
        assert len(result) == 1
        assert result[0].value == 2  # type: ignore[attr-defined]


def test_run_stage_replay_with_filtering(tmp_path: Path) -> None:
    """Test replay with stages that filter tasks."""
    # Create task batches with multiple values
    serializer = PickleTaskSerializer()
    serializer.save(tmp_path / "batch_000.pkl", [TestTask(value=1), TestTask(value=2), TestTask(value=3)])

    # Execute with direct executor: increment then filter odd values
    # (1+1)=2 (filtered), (2+1)=3 (kept), (3+1)=4 (filtered)
    executor = DirectStageExecutor()
    result = run_stage_replay(
        stages=[IncrementStage(), FilterStage()],
        path=tmp_path,
        limit=1,
        executor=executor,
        serializer=serializer,
        init_ray=False,
    )

    # Only odd value (3) should remain
    assert len(result) == 1
    assert result[0].value == 3  # type: ignore[attr-defined]


def test_run_stage_replay_empty_result(tmp_path: Path) -> None:
    """Test replay when stages filter out all tasks."""
    # Create task batches with even values
    serializer = PickleTaskSerializer()
    serializer.save(tmp_path / "batch_000.pkl", [TestTask(value=2), TestTask(value=4)])

    # Execute with filter that removes all even values
    executor = DirectStageExecutor()
    result = run_stage_replay(
        stages=[FilterStage()],
        path=tmp_path,
        limit=1,
        executor=executor,
        serializer=serializer,
        init_ray=False,
    )

    # All tasks should be filtered out
    assert len(result) == 0


def test_direct_stage_executor_calls_setup_and_destroy() -> None:
    """Test that DirectStageExecutor calls setup and destroy methods."""

    class SetupTrackingStage(CuratorStage):
        """Stage that tracks lifecycle method calls."""

        def __init__(self) -> None:
            """Initialize tracking flags."""
            self.setup_on_node_called = False
            self.stage_setup_called = False
            self.destroy_called = False

        def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
            """Track setup_on_node call."""
            self.setup_on_node_called = True

        def stage_setup(self) -> None:
            """Track stage_setup call."""
            self.stage_setup_called = True

        def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
            """Process data."""
            return tasks

        def destroy(self) -> None:
            """Track destroy call."""
            self.destroy_called = True

    stage = SetupTrackingStage()
    executor = DirectStageExecutor()
    node_info = NodeInfo(node_id="test")
    worker_metadata = WorkerMetadata.make_dummy()

    task_batches = [[TestTask(value=1)]]
    executor.execute_stage(stage, task_batches, node_info, worker_metadata)  # type: ignore[arg-type]

    # Verify all lifecycle methods were called
    assert stage.setup_on_node_called
    assert stage.stage_setup_called
    assert stage.destroy_called


# ============================================================================
# Tests for RayStageExecutor with Mock
# ============================================================================


def test_ray_stage_executor_basic_execution() -> None:
    """Test RayStageExecutor with mock Ray operations."""
    # Create a real stage to process tasks
    stage = IncrementStage()

    # Mock the stage runner
    mock_runner = MagicMock()
    mock_runner.setup_on_node.remote.return_value = "setup_on_node_ref"
    mock_runner.stage_setup.remote.return_value = "stage_setup_ref"
    mock_runner.destroy.remote.return_value = "destroy_ref"

    # Mock process_data to actually process tasks through the stage
    def mock_process_data(tasks: list[PipelineTask]) -> str:
        """Mock that processes tasks through the real stage."""
        mock_runner._last_result = stage.process_data(tasks)
        return "process_data_ref"

    mock_runner.process_data.remote.side_effect = mock_process_data

    # Mock ray.get to return the appropriate results
    def mock_ray_get(ref: str) -> Any:  # noqa: ANN401
        if ref == "process_data_ref":
            return mock_runner._last_result
        return None

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.StageRunner") as mock_stage_runner_cls,
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.remote", return_value=mock_stage_runner_cls),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.get", side_effect=mock_ray_get),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.kill") as mock_ray_kill,
    ):
        mock_stage_runner_cls.options.return_value.remote.return_value = mock_runner

        executor = RayStageExecutor()
        result = executor.execute_stage(
            stage=IncrementStage(),
            task_batches=[[TestTask(value=1)], [TestTask(value=5)]],
            node_info=NodeInfo(node_id="test"),
            worker_metadata=WorkerMetadata.make_dummy(),
        )

        # Verify correct execution through the stage
        assert len(result) == 2
        assert result[0][0].value == 2  # type: ignore[attr-defined]
        assert result[1][0].value == 6  # type: ignore[attr-defined]

        # Verify stage runner methods were called
        mock_runner.setup_on_node.remote.assert_called_once()
        mock_runner.stage_setup.remote.assert_called_once()
        assert mock_runner.process_data.remote.call_count == 2
        mock_runner.destroy.remote.assert_called_once()
        mock_ray_kill.assert_called_once_with(mock_runner)


def test_ray_stage_executor_resource_allocation() -> None:
    """Test that RayStageExecutor correctly passes resource requirements."""

    class CustomResourceStage(CuratorStage):
        """Stage with custom resource requirements."""

        @property
        def required_resources(self) -> Resources:
            """Return custom resource requirements."""
            return Resources(cpus=4.5, gpus=2.0)

        def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
            """Process data."""
            return tasks

    mock_runner = MagicMock()
    mock_options = MagicMock()

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.StageRunner") as mock_stage_runner_cls,
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.remote", return_value=mock_stage_runner_cls),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.get"),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.kill"),
    ):
        mock_stage_runner_cls.options.return_value = mock_options
        mock_options.remote.return_value = mock_runner

        executor = RayStageExecutor()
        executor.execute_stage(
            stage=CustomResourceStage(),
            task_batches=[[TestTask(value=1)]],
            node_info=NodeInfo(node_id="test"),
            worker_metadata=WorkerMetadata.make_dummy(),
        )

        # Verify resources were passed correctly to options()
        mock_stage_runner_cls.options.assert_called_once()
        call_kwargs = mock_stage_runner_cls.options.call_args[1]
        assert call_kwargs["num_cpus"] == 4.5
        assert call_kwargs["num_gpus"] == 2.0


def test_ray_stage_executor_multiple_batches() -> None:
    """Test RayStageExecutor with multiple task batches."""
    stage = DoubleStage()
    mock_runner = MagicMock()

    def mock_process_data(tasks: list[PipelineTask]) -> str:
        """Mock that processes tasks through the real stage."""
        mock_runner._last_result = stage.process_data(tasks)
        return "process_data_ref"

    mock_runner.process_data.remote.side_effect = mock_process_data

    def mock_ray_get(ref: str) -> Any:  # noqa: ANN401
        if ref == "process_data_ref":
            return mock_runner._last_result
        return None

    task_batches = [
        [TestTask(value=1), TestTask(value=2)],
        [TestTask(value=3)],
        [TestTask(value=4), TestTask(value=5), TestTask(value=6)],
    ]

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.StageRunner") as mock_stage_runner_cls,
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.remote", return_value=mock_stage_runner_cls),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.get", side_effect=mock_ray_get),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.kill"),
    ):
        mock_stage_runner_cls.options.return_value.remote.return_value = mock_runner

        executor = RayStageExecutor()
        result = executor.execute_stage(
            stage=DoubleStage(),
            task_batches=task_batches,  # type: ignore[arg-type]
            node_info=NodeInfo(node_id="test"),
            worker_metadata=WorkerMetadata.make_dummy(),
        )

        # Verify all batches processed correctly
        assert len(result) == 3
        assert len(result[0]) == 2
        assert result[0][0].value == 2  # type: ignore[attr-defined]
        assert result[0][1].value == 4  # type: ignore[attr-defined]
        assert len(result[1]) == 1
        assert result[1][0].value == 6  # type: ignore[attr-defined]
        assert len(result[2]) == 3
        assert result[2][0].value == 8  # type: ignore[attr-defined]
        assert result[2][1].value == 10  # type: ignore[attr-defined]
        assert result[2][2].value == 12  # type: ignore[attr-defined]


def test_ray_stage_executor_empty_batches() -> None:
    """Test RayStageExecutor handles empty result batches."""
    stage = FilterStage()
    mock_runner = MagicMock()

    def mock_process_data(tasks: list[PipelineTask]) -> str:
        """Mock that processes tasks through the real stage."""
        mock_runner._last_result = stage.process_data(tasks)
        return "process_data_ref"

    mock_runner.process_data.remote.side_effect = mock_process_data

    def mock_ray_get(ref: str) -> Any:  # noqa: ANN401
        if ref == "process_data_ref":
            return mock_runner._last_result
        return None

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.StageRunner") as mock_stage_runner_cls,
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.remote", return_value=mock_stage_runner_cls),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.get", side_effect=mock_ray_get),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.kill"),
    ):
        mock_stage_runner_cls.options.return_value.remote.return_value = mock_runner

        executor = RayStageExecutor()
        result = executor.execute_stage(
            stage=FilterStage(),  # Filters out even values
            task_batches=[[TestTask(value=2), TestTask(value=4)]],  # All even
            node_info=NodeInfo(node_id="test"),
            worker_metadata=WorkerMetadata.make_dummy(),
        )

        # All tasks filtered out, should get empty batch
        assert len(result) == 1
        assert len(result[0]) == 0


def test_ray_stage_executor_conda_env_default() -> None:
    """Test that default conda environment is used when stage doesn't specify one."""

    class NoEnvStage(CuratorStage):
        """Stage without conda_env_name."""

        @property
        def conda_env_name(self) -> None:
            """Return None for conda env."""
            return None

        def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
            """Process data."""
            return tasks

    mock_runner = MagicMock()

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.StageRunner") as mock_stage_runner_cls,
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.remote", return_value=mock_stage_runner_cls),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.get"),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.kill"),
        patch("cosmos_curator.core.utils.misc.stage_replay.PixiRuntimeEnv") as mock_pixi_env,
    ):
        mock_stage_runner_cls.options.return_value.remote.return_value = mock_runner

        executor = RayStageExecutor()
        executor.execute_stage(
            stage=NoEnvStage(),
            task_batches=[[TestTask(value=1)]],
            node_info=NodeInfo(node_id="test"),
            worker_metadata=WorkerMetadata.make_dummy(),
        )

        # Verify PixiRuntimeEnv was created with "default"
        mock_pixi_env.assert_called_once_with("default")


def test_ray_stage_executor_calls_methods_in_order() -> None:
    """Test that RayStageExecutor calls stage runner methods in correct order."""
    call_order: list[str] = []
    mock_runner = MagicMock()

    def track_setup_on_node(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401, ARG001
        call_order.append("setup_on_node")
        return "setup_on_node_ref"

    def track_stage_setup(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401, ARG001
        call_order.append("stage_setup")
        return "stage_setup_ref"

    def track_process_data(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401, ARG001
        call_order.append("process_data")
        mock_runner._last_result = args[0]  # Return the tasks unchanged
        return "process_data_ref"

    def track_destroy(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401, ARG001
        call_order.append("destroy")
        return "destroy_ref"

    mock_runner.setup_on_node.remote.side_effect = track_setup_on_node
    mock_runner.stage_setup.remote.side_effect = track_stage_setup
    mock_runner.process_data.remote.side_effect = track_process_data
    mock_runner.destroy.remote.side_effect = track_destroy

    def mock_ray_get(ref: str) -> Any:  # noqa: ANN401
        if ref == "process_data_ref":
            return mock_runner._last_result
        return None

    with (
        patch("cosmos_curator.core.utils.misc.stage_replay.StageRunner") as mock_stage_runner_cls,
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.remote", return_value=mock_stage_runner_cls),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.get", side_effect=mock_ray_get),
        patch("cosmos_curator.core.utils.misc.stage_replay.ray.kill"),
    ):
        mock_stage_runner_cls.options.return_value.remote.return_value = mock_runner

        executor = RayStageExecutor()
        executor.execute_stage(
            stage=IncrementStage(),
            task_batches=[[TestTask(value=1)], [TestTask(value=2)]],
            node_info=NodeInfo(node_id="test"),
            worker_metadata=WorkerMetadata.make_dummy(),
        )

        # Verify correct call order
        expected_order = ["setup_on_node", "stage_setup", "process_data", "process_data", "destroy"]
        assert call_order == expected_order


def test_stage_runner_smoke() -> None:
    """Smoke test for StageRunner - ensure all methods can be called without errors."""
    # Create a mock stage that does nothing
    mock_stage = MagicMock(spec=CuratorStage)
    mock_stage.process_data.return_value = [TestTask(value=1)]

    # Create StageRunner with mock stage
    stage_runner = StageRunner(mock_stage)

    # Verify the stage is stored
    assert stage_runner.stage is mock_stage

    # Call setup_on_node - should not blow up
    node_info = NodeInfo(node_id="test_node")
    worker_metadata = WorkerMetadata.make_dummy()
    stage_runner.setup_on_node(node_info, worker_metadata)
    mock_stage.setup_on_node.assert_called_once_with(node_info, worker_metadata)

    # Call stage_setup - should not blow up
    stage_runner.stage_setup()
    mock_stage.stage_setup.assert_called_once()

    # Call process_data - should not blow up
    tasks = [TestTask(value=1), TestTask(value=2)]
    result = stage_runner.process_data(tasks)
    mock_stage.process_data.assert_called_once_with(tasks)
    assert result == [TestTask(value=1)]

    # Call destroy - should not blow up
    stage_runner.destroy()
    mock_stage.destroy.assert_called_once()
