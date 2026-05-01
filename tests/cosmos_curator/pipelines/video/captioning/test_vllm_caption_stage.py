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
"""Test vllm_caption_stage.py."""

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import attrs
import pytest

from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.models.vllm_model_ids import _VLLM_MODELS
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import _scatter_captions
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    TokenCounts,
    Video,
    VllmConfig,
    Window,
    WindowConfig,
)

if conda_utils.is_running_in_env("unified"):
    import torch

    from cosmos_curator.models.vllm_interface import _VLLM_PLUGINS, VllmWindowResult
    from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import (
        VllmCaptionStage,
        VllmModelInterface,
        VllmPrepStage,
        _free_vllm_inputs,
        _get_stage2_prompts,
        _get_windows_from_tasks,
        _scatter_captions,
    )
    from cosmos_curator.pipelines.video.utils.data_model import get_video_from_task

    VALID_VARIANTS = list(_VLLM_PLUGINS.keys())
else:

    @attrs.define
    class VllmWindowResult:
        """Fallback result shape used when the unified environment is unavailable."""

        text: str
        finish_reason: str | None
        token_counts: TokenCounts

    VALID_VARIANTS = []


# Test UUIDs for deterministic testing
UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
UUID_2 = UUID("00000000-0000-0000-0000-000000000002")
UUID_3 = UUID("00000000-0000-0000-0000-000000000003")


@pytest.mark.env("unified")
def test_get_video_from_task_success() -> None:
    """Test get_video_from_task."""
    task = SplitPipeTask(session_id="test-session", video=Video(input_video=Path("test.mp4")))
    video = get_video_from_task(task)
    assert video.input_video == Path("test.mp4")


@pytest.mark.env("unified")
def test_get_video_from_task_fail() -> None:
    """Test get_video_from_task."""
    task = 10
    with pytest.raises(TypeError, match=r".*"):
        get_video_from_task(task)  # type: ignore[type-var]


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("config_variant", "raises"),
    [(k, nullcontext()) for k in _VLLM_MODELS] + [("_fail_model", pytest.raises(ValueError, match=r".*"))],
)
def test_vllm_model_interface_model_id_names(config_variant: str, raises: AbstractContextManager[Any]) -> None:
    """Validate model_id_names are strings for each configured plugin variant."""
    vllm_config = VllmConfig(model_variant=config_variant)
    vllm_model_interface = VllmModelInterface(vllm_config)

    with raises:
        model_id_names = vllm_model_interface.model_id_names
        for model_id_name in model_id_names:
            assert isinstance(model_id_name, str)


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("tasks", "expected_windows", "raises"),
    [
        # Empty tasks list
        ([], [], nullcontext()),
        # Single task with no clips
        ([SplitPipeTask(session_id="test-session", video=Video(input_video=Path("test.mp4")))], [], nullcontext()),
        # Single task with clip but no windows
        (
            [
                SplitPipeTask(
                    session_id="test-session",
                    video=Video(
                        input_video=Path("test.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_1,
                                source_video="test.mp4",
                                span=(0.0, 1.0),
                            )
                        ],
                    ),
                )
            ],
            [],
            nullcontext(),
        ),
        # Single task with clip and windows
        (
            [
                SplitPipeTask(
                    session_id="test-session",
                    video=Video(
                        input_video=Path("test.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_1,
                                source_video="test.mp4",
                                span=(0.0, 1.0),
                                windows=[
                                    Window(
                                        start_frame=0,
                                        end_frame=10,
                                    ),
                                    Window(
                                        start_frame=10,
                                        end_frame=20,
                                    ),
                                ],
                            )
                        ],
                    ),
                )
            ],
            [
                (Window(start_frame=0, end_frame=10), UUID_1),
                (Window(start_frame=10, end_frame=20), UUID_1),
            ],
            nullcontext(),
        ),
        # Multiple tasks with mixed scenarios
        (
            [
                SplitPipeTask(
                    session_id="test-session",
                    video=Video(
                        input_video=Path("test1.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_1,
                                source_video="test1.mp4",
                                span=(0.0, 1.0),
                                windows=[
                                    Window(
                                        start_frame=0,
                                        end_frame=10,
                                    )
                                ],
                            )
                        ],
                    ),
                ),
                SplitPipeTask(
                    session_id="test-session",
                    video=Video(
                        input_video=Path("test2.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_2,
                                source_video="test2.mp4",
                                span=(0.0, 1.0),
                            ),  # No windows
                            Clip(
                                uuid=UUID_3,
                                source_video="test2.mp4",
                                span=(1.0, 2.0),
                                windows=[
                                    Window(
                                        start_frame=20,
                                        end_frame=30,
                                    )
                                ],
                            ),
                        ],
                    ),
                ),
            ],
            [
                (Window(start_frame=0, end_frame=10), UUID_1),
                (Window(start_frame=20, end_frame=30), UUID_3),
            ],
            nullcontext(),
        ),
    ],
)
def test_get_windows_from_tasks(
    tasks: list[Any], expected_windows: list[tuple[Window, UUID]], raises: AbstractContextManager[Any]
) -> None:
    """Test _get_windows_from_tasks function."""
    with raises:
        windows, clip_uuids = _get_windows_from_tasks(tasks)
        assert len(windows) == len(expected_windows)
        assert len(clip_uuids) == len(expected_windows)
        for (actual_window, actual_clip_uuid), (expected_window, expected_clip_uuid) in zip(
            zip(windows, clip_uuids, strict=True), expected_windows, strict=True
        ):
            assert actual_clip_uuid == str(expected_clip_uuid)
            assert actual_window.start_frame == expected_window.start_frame
            assert actual_window.end_frame == expected_window.end_frame


@pytest.mark.env("unified")
@pytest.mark.parametrize("keep_mp4", [False, True])
def test_free_vllm_inputs_clears_inputs_and_optionally_mp4(*, keep_mp4: bool) -> None:
    """Validate model inputs are removed and mp4_bytes handled per flag."""
    model_variant = "test_variant"
    other_variant = "other_variant"
    w1 = Window(
        start_frame=0,
        end_frame=10,
        mp4_bytes=b"a",
        model_input={model_variant: {"x": 1}, other_variant: {"z": 3}},
    )
    w2 = Window(
        start_frame=10,
        end_frame=20,
        mp4_bytes=b"b",
        model_input={model_variant: {"y": 2}, other_variant: {"k": 4}},
    )

    original_bytes = [w1.mp4_bytes, w2.mp4_bytes]
    _free_vllm_inputs([w1, w2], model_variant, keep_mp4=keep_mp4)

    for idx, w in enumerate([w1, w2]):
        assert model_variant not in w.model_input
        # Ensure other variant remains untouched
        assert other_variant in w.model_input
        assert set(w.model_input.keys()) == {other_variant}
        if keep_mp4:
            assert w.mp4_bytes is original_bytes[idx]
        else:
            assert w.mp4_bytes.resolve() is None


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("windows_count", "frames_count", "raises"),
    [
        (2, 2, nullcontext()),
        (1, 2, pytest.raises(ValueError, match=r".*")),
        (3, 1, pytest.raises(ValueError, match=r".*")),
    ],
)
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
@patch("cosmos_curator.pipelines.video.captioning.vllm_caption_stage.windowing_utils.make_windows_for_video")
@patch("cosmos_curator.models.vllm_interface.make_model_inputs")
def test_prep_windows_model_input_assignment(  # noqa: PLR0913
    mock_make_model_inputs: MagicMock,
    mock_make_windows: MagicMock,
    model_variant: str,
    windows_count: int,
    frames_count: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Validate VllmPrepStage._prep_windows assigns inputs and enforces strict zipping."""
    config = VllmConfig(model_variant=model_variant)
    window_config = WindowConfig()
    stage = VllmPrepStage(config, window_config, keep_mp4=False)
    # Inject a fake processor since stage_setup isn't called here
    stage._processor = MagicMock()  # type: ignore[attr-defined]

    prompt = "test prompt"
    video = Video(input_video=Path("test.mp4"))

    # Create test data returned by windowing util
    windows = [Window(start_frame=0, end_frame=frames_count) for _ in range(windows_count)]
    frames = [torch.randn(frames_count, 3, 224, 224) for _ in range(frames_count)]
    mock_make_windows.return_value = (windows, frames)
    mock_make_model_inputs.return_value = [{"test": "data"} for _ in range(frames_count)]

    with raises:
        stage._prep_windows(video, prompt)

        for window in windows:
            llm_input = window.model_input.get(model_variant)
            assert isinstance(llm_input, dict)


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
@patch("cosmos_curator.pipelines.video.captioning.vllm_caption_stage.windowing_utils.make_windows_for_video")
def test_prep_windows_raises_without_processor(mock_make_windows: MagicMock, model_variant: str) -> None:
    """_prep_windows should raise RuntimeError if self._processor is not set."""
    config = VllmConfig(model_variant=model_variant)
    stage = VllmPrepStage(config, WindowConfig(), keep_mp4=False)
    # Do NOT set stage._processor here

    video = Video(input_video=Path("test.mp4"))
    windows = [Window(start_frame=0, end_frame=10)]
    frames: list[object] = [object()]
    mock_make_windows.return_value = (windows, frames)

    with pytest.raises(RuntimeError, match=r".*processor.*"):
        stage._prep_windows(video, "prompt")


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("stage2_prompt_text", "stage2_caption"),
    [
        ("test prompt", True),
        (None, False),
        (None, True),
    ],
)
def test_get_stage2_prompts(stage2_prompt_text: str | None, *, stage2_caption: bool) -> None:
    """Test _get_stage2_prompts."""
    vllm_config = VllmConfig(
        model_variant="test_variant", stage2_caption=stage2_caption, stage2_prompt_text=stage2_prompt_text
    )
    num_windows = 10
    stage2_prompts = _get_stage2_prompts(vllm_config, num_windows=num_windows)

    assert len(stage2_prompts) == num_windows

    if stage2_caption:
        for prompt in stage2_prompts:
            if stage2_prompt_text is None:
                assert isinstance(prompt, str)
            else:
                assert prompt == stage2_prompt_text
    else:
        for prompt in stage2_prompts:
            assert prompt is None


@pytest.mark.env("unified")
@pytest.mark.parametrize("verbose", [True, False])
def test_scatter_captions(*, verbose: bool) -> None:
    """Test _scatter_captions."""
    windows = [Window(start_frame=0, end_frame=10), Window(start_frame=10, end_frame=20)]
    results = [
        VllmWindowResult(text="caption 1", finish_reason="stop", token_counts=TokenCounts(10, 5)),
        VllmWindowResult(text="caption 2", finish_reason="stop", token_counts=TokenCounts(8, 3)),
    ]
    clip_uuids = ["clip_uuid_1", "clip_uuid_2"]
    model_variant = "test_variant"
    _scatter_captions(windows, results, clip_uuids, model_variant, verbose=verbose)
    for window, result in zip(windows, results, strict=True):
        assert window.caption[model_variant] == result.text
        assert window.token_counts[model_variant] == result.token_counts


@pytest.mark.parametrize(
    ("result", "expect_caption", "expected_status", "expected_reason"),
    [
        (VllmWindowResult("A well-formed caption.", "stop", TokenCounts()), True, "success", None),
        (VllmWindowResult("Trimmed caption.  ", "length", TokenCounts()), True, "truncated", None),
        (VllmWindowResult("", "length", TokenCounts()), False, "error", "exception"),
        (VllmWindowResult("", "stop", TokenCounts()), False, "error", "exception"),
        (VllmWindowResult(VLLM_UNKNOWN_CAPTION, "length", TokenCounts()), False, "error", "exception"),
    ],
    ids=["success", "truncated", "empty_length_error", "empty_error", "sentinel_error"],
)
def test_scatter_captions_sets_status(
    result: VllmWindowResult, *, expect_caption: bool, expected_status: str, expected_reason: str | None
) -> None:
    """_scatter_captions writes caption_status and caption_failure_reason for each outcome."""
    window = Window(start_frame=0, end_frame=10)
    _scatter_captions([window], [result], ["clip_1"], "qwen", verbose=False)
    if expect_caption:
        assert window.caption["qwen"] == result.text.strip()
    else:
        assert "qwen" not in window.caption
    assert window.caption_status == expected_status
    assert window.caption_failure_reason == expected_reason


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("copy_weights_to"),
    [
        (None),
        ("custom_path"),
    ],
    ids=["no_copy_weights_to", "with_copy_weights_to"],
)
def test_setup_on_node_copies_weights(
    tmp_path: Path,
    copy_weights_to: str | None,
) -> None:
    """Test VllmCaptionStage.stage_setup_on_node copies model weights correctly.

    Args:
        tmp_path: Pytest temporary directory fixture.
        copy_weights_to: The copy_weights_to config value (None or "custom_path").

    """
    should_copy = copy_weights_to is not None
    # Resolve copy_weights_to path if provided
    resolved_copy_weights_to = tmp_path / copy_weights_to if copy_weights_to is not None else None

    # Create mock source directory with a single file to validate
    mock_source_dir = tmp_path / "mock_source"
    mock_source_dir.mkdir()
    (mock_source_dir / "model.bin").write_bytes(b"mock_model_data")
    (mock_source_dir / "config.json").write_text('{"test": "config"}')

    # Create VllmConfig with copy_weights_to
    vllm_config = VllmConfig(
        model_variant="qwen",
        copy_weights_to=resolved_copy_weights_to,
    )

    stage = VllmCaptionStage(vllm_config=vllm_config)

    # Mock get_local_dir_for_weights_name to return our mock source directory
    with (
        patch(
            "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.model_utils.get_local_dir_for_weights_name"
        ) as mock_get_local_dir,
        patch(
            "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.vllm_model",
            return_value=MagicMock(),
        ),
        patch(
            "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.gpu_stage_startup",
            return_value=None,
        ),
    ):
        mock_get_local_dir.return_value = mock_source_dir

        # Call stage_setup_on_node (uses real copy_model_weights)
        stage.stage_setup_on_node()

        # Verify the results based on whether copying should occur
        if should_copy:
            assert resolved_copy_weights_to is not None
            # Verify files were actually copied to destination
            model_id_names = stage._model.model_id_names
            for model_id in model_id_names:
                dest_dir = resolved_copy_weights_to / model_id
                assert dest_dir.exists(), f"Destination directory not created for {model_id}"
                assert (dest_dir / "model.bin").exists(), f"model.bin not copied for {model_id}"
                assert (dest_dir / "model.bin").read_bytes() == b"mock_model_data"
                assert (dest_dir / "config.json").exists(), f"config.json not copied for {model_id}"
                assert (dest_dir / "config.json").read_text() == '{"test": "config"}'
        else:
            # Verify no files were copied (copy_weights_to is None)
            assert resolved_copy_weights_to is None

            # Verify filesystem: no destination directories should have been created
            model_id_names = stage._model.model_id_names
            for model_id in model_id_names:
                # Check that no directory exists in tmp_path with this model_id
                # (we only expect mock_source to exist)
                potential_dest = tmp_path / model_id
                assert not potential_dest.exists(), f"Unexpected directory created: {potential_dest}"

            # Verify only the mock_source_dir exists in tmp_path (no other dirs created)
            existing_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
            assert len(existing_dirs) == 1, f"Expected only mock_source, found: {existing_dirs}"
            assert existing_dirs[0] == mock_source_dir


@pytest.mark.env("unified")
def test_setup_on_node_raises_when_source_directory_missing(tmp_path: Path) -> None:
    """Test VllmCaptionStage.stage_setup_on_node raises error when source directory doesn't exist."""
    # Create VllmConfig with copy_weights_to
    copy_weights_to = tmp_path / "custom_weights"
    vllm_config = VllmConfig(
        model_variant="qwen",
        copy_weights_to=copy_weights_to,
    )

    stage = VllmCaptionStage(vllm_config=vllm_config)

    # Mock get_local_dir_for_weights_name to return non-existent directory
    # This should cause stage_setup_on_node to raise FileNotFoundError
    nonexistent_source_dir = tmp_path / "nonexistent_source"

    with patch(
        "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.model_utils.get_local_dir_for_weights_name"
    ) as mock_get_local_dir:
        mock_get_local_dir.return_value = nonexistent_source_dir

        # Verify that stage_setup_on_node raises FileNotFoundError
        with pytest.raises(FileNotFoundError, match=r".*"):
            stage.stage_setup_on_node()


@pytest.mark.env("unified")
def test_setup_on_node_handles_copy_failure(tmp_path: Path) -> None:
    """Test VllmCaptionStage.stage_setup_on_node handles copy failures gracefully."""
    # Create VllmConfig with copy_weights_to
    copy_weights_to = tmp_path / "custom_weights"
    vllm_config = VllmConfig(
        model_variant="qwen",
        copy_weights_to=copy_weights_to,
    )

    stage = VllmCaptionStage(vllm_config=vllm_config)

    # Create a mock source directory
    mock_source_dir = tmp_path / "mock_source"
    mock_source_dir.mkdir()

    with (
        patch(
            "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.model_utils.get_local_dir_for_weights_name"
        ) as mock_get_local_dir,
        patch(
            "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.model_utils.copy_model_weights"
        ) as mock_copy_weights,
        patch(
            "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.vllm_model",
            return_value=MagicMock(),
        ),
        patch(
            "cosmos_curator.pipelines.video.captioning.vllm_caption_stage.gpu_stage_startup",
            return_value=None,
        ),
    ):
        mock_get_local_dir.return_value = mock_source_dir
        # Simulate a copy failure (e.g., permission denied, disk full, etc.)
        mock_copy_weights.side_effect = OSError("Permission denied")

        # Verify that stage_setup_on_node completes without raising (swallows the exception)
        stage.stage_setup_on_node()

        # Verify copy_model_weights was called (but failed)
        model_id_names = stage._model.model_id_names
        assert mock_copy_weights.call_count == len(model_id_names)

        # Verify no files were actually copied (destination should not exist or be empty)
        for model_id in model_id_names:
            dest_dir = copy_weights_to / model_id
            # Either the directory doesn't exist or it exists but is empty
            if dest_dir.exists():
                assert not any(dest_dir.iterdir()), f"Expected {dest_dir} to be empty after failed copy"
