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

import threading
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import attrs
import psutil
import pytest

from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.models.vllm_model_ids import _VLLM_MODELS
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.video.captioning import vllm_caption_stage
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


def _make_caption_stage_task(*, use_filter_windows: bool = False) -> SplitPipeTask:
    """Create a minimal task whose selected windows have vLLM model input."""
    window = Window(start_frame=0, end_frame=10, model_input={"qwen": {"prompt": "input"}})
    clip = Clip(uuid=UUID_1, source_video="test.mp4", span=(0.0, 1.0))
    if use_filter_windows:
        clip.filter_windows = [window]
    else:
        clip.windows = [window]
    video = Video(input_video=Path("test.mp4"), clips=[clip])
    return SplitPipeTask(session_id="test-session", video=video)


def _process_caption_stage_with_quality_patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    caption_quality_flags_enabled: bool = True,
    use_filter_windows: bool = False,
) -> tuple[MagicMock, SplitPipeTask]:
    """Run VllmCaptionStage.process_data with mocked inference and return the quality mock."""
    task = _make_caption_stage_task(use_filter_windows=use_filter_windows)
    stage = vllm_caption_stage.VllmCaptionStage(
        vllm_config=VllmConfig(model_variant="qwen"),
        caption_quality_flags_enabled=caption_quality_flags_enabled,
        use_filter_windows=use_filter_windows,
    )
    stage._llm = object()  # type: ignore[attr-defined]
    stage._sampling_params = object()  # type: ignore[attr-defined]
    stage._processor = object()  # type: ignore[attr-defined]

    monkeypatch.setattr(
        vllm_caption_stage,
        "vllm_caption",
        lambda *_args, **_kwargs: [VllmWindowResult("a useful caption text", "stop", TokenCounts())],
        raising=False,
    )
    quality_mock = MagicMock()
    monkeypatch.setattr(vllm_caption_stage, "apply_caption_quality_flags", quality_mock)

    stage.process_data([task])
    return quality_mock, task


def test_process_data_applies_caption_quality_flags_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled subject-caption runs should invoke caption quality flagging."""
    quality_mock, task = _process_caption_stage_with_quality_patch(monkeypatch)

    quality_mock.assert_called_once()
    window_groups, model_variant = quality_mock.call_args.args
    assert model_variant == "qwen"
    assert len(window_groups) == 1
    assert len(window_groups[0]) == 1
    # Object identity: quality flagging mutates the actual Window, not a copy.
    assert window_groups[0][0] is task.video.clips[0].windows[0]


def test_process_data_skips_caption_quality_flags_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled caption quality runs should skip flagging."""
    quality_mock, _ = _process_caption_stage_with_quality_patch(monkeypatch, caption_quality_flags_enabled=False)

    quality_mock.assert_not_called()


def test_process_data_skips_caption_quality_flags_for_filter_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filter-window vLLM runs should skip subject-caption quality flagging."""
    quality_mock, _ = _process_caption_stage_with_quality_patch(monkeypatch, use_filter_windows=True)

    quality_mock.assert_not_called()


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


# ---------------------------------------------------------------------------
# EngineCore liveness / orphan-reaper / watchdog helpers.
#
# These exercise the actor-death recovery hardening added alongside the
# xenna 0.4.3 bump. They are intentionally CPU-only (no ``env`` marker): every
# path is driven through mocked ``psutil`` / ``threading`` primitives, so no
# GPU, vLLM, or real subprocess is required.
# ---------------------------------------------------------------------------


def _make_watchdog_stage() -> "vllm_caption_stage.VllmCaptionStage":
    """Construct a bare ``VllmCaptionStage`` for liveness/watchdog unit tests.

    Construction is cheap on CPU (matches the other non-env tests above) and the
    watchdog/liveness helpers only touch ``_watchdog_thread``, ``_watchdog_stop_event``,
    and ``_llm`` plus the (mocked) ``psutil`` surface.
    """
    return vllm_caption_stage.VllmCaptionStage(vllm_config=VllmConfig(model_variant="qwen"))


def _fake_proc(cmdline: list[str], status: str = psutil.STATUS_RUNNING) -> MagicMock:
    """Return a ``psutil.Process``-like mock with the given cmdline/status."""
    proc = MagicMock()
    proc.cmdline.return_value = cmdline
    proc.status.return_value = status
    return proc


def test_is_live_engine_core_detects_running_engine_core() -> None:
    """A running process whose cmdline mentions EngineCore is reported live."""
    proc = _fake_proc(["python", "-c", "from vllm ... VLLM::EngineCore"])
    assert vllm_caption_stage.VllmCaptionStage._is_live_engine_core(proc) is True


def test_is_live_engine_core_ignores_non_engine_core() -> None:
    """A process without EngineCore in its cmdline is not a live EngineCore."""
    proc = _fake_proc(["bash", "-c", "sleep"])
    assert vllm_caption_stage.VllmCaptionStage._is_live_engine_core(proc) is False


def test_is_live_engine_core_treats_zombie_as_dead() -> None:
    """A zombie/dead EngineCore must not count as live."""
    zombie = _fake_proc(["VLLM::EngineCore"], status=psutil.STATUS_ZOMBIE)
    assert vllm_caption_stage.VllmCaptionStage._is_live_engine_core(zombie) is False


def test_is_live_engine_core_handles_vanished_process() -> None:
    """If the process vanishes mid-inspection, treat it as not-live (no raise)."""
    proc = MagicMock()
    proc.cmdline.side_effect = psutil.NoSuchProcess(1234)
    assert vllm_caption_stage.VllmCaptionStage._is_live_engine_core(proc) is False


def test_engine_core_is_alive_true_when_child_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """At least one live EngineCore child => alive."""
    stage = _make_watchdog_stage()
    parent = MagicMock()
    parent.children.return_value = [_fake_proc(["bash"]), _fake_proc(["VLLM::EngineCore"])]
    monkeypatch.setattr(vllm_caption_stage.psutil, "Process", lambda: parent)
    assert stage._engine_core_is_alive() is True


def test_engine_core_is_alive_false_when_no_engine_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Children present but none are EngineCore => dead."""
    stage = _make_watchdog_stage()
    parent = MagicMock()
    parent.children.return_value = [_fake_proc(["bash"]), _fake_proc(["python", "other"])]
    monkeypatch.setattr(vllm_caption_stage.psutil, "Process", lambda: parent)
    assert stage._engine_core_is_alive() is False


def test_engine_core_is_alive_false_on_no_such_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """If our own process record vanishes, report dead."""
    stage = _make_watchdog_stage()
    parent = MagicMock()
    parent.children.side_effect = psutil.NoSuchProcess(1)
    monkeypatch.setattr(vllm_caption_stage.psutil, "Process", lambda: parent)
    assert stage._engine_core_is_alive() is False


def test_engine_core_is_alive_fails_open_on_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """When psutil cannot enumerate children, assume alive (no false-positive recovery)."""
    stage = _make_watchdog_stage()
    parent = MagicMock()
    parent.children.side_effect = psutil.AccessDenied()
    monkeypatch.setattr(vllm_caption_stage.psutil, "Process", lambda: parent)
    assert stage._engine_core_is_alive() is True


def test_in_ray_actor_context_false_outside_actor() -> None:
    """Outside a Ray actor (i.e. under pytest) the guard must report False."""
    assert vllm_caption_stage.VllmCaptionStage._in_ray_actor_context() is False


def test_kill_orphan_engine_cores_skips_when_actor_is_pid_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the actor itself is PID 1, the sweep is skipped (can't disambiguate orphans)."""
    stage = _make_watchdog_stage()
    monkeypatch.setattr(vllm_caption_stage.os, "getpid", lambda: 1)
    process_iter = MagicMock()
    monkeypatch.setattr(vllm_caption_stage.psutil, "process_iter", process_iter)
    stage._kill_orphan_engine_cores()
    process_iter.assert_not_called()


def test_kill_orphan_engine_cores_kills_only_reparented_orphans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only EngineCore procs reparented to PID 1 are killed; live siblings are untouched."""
    stage = _make_watchdog_stage()
    monkeypatch.setattr(vllm_caption_stage.os, "getpid", lambda: 42)

    orphan = MagicMock()
    orphan.info = {"pid": 100, "ppid": 1, "cmdline": ["python", "VLLM::EngineCore"]}
    live_sibling = MagicMock()  # ppid points at a live actor, not init
    live_sibling.info = {"pid": 101, "ppid": 42, "cmdline": ["python", "VLLM::EngineCore"]}
    unrelated_orphan = MagicMock()  # reparented to init but not an EngineCore
    unrelated_orphan.info = {"pid": 102, "ppid": 1, "cmdline": ["bash", "-c", "sleep"]}

    monkeypatch.setattr(
        vllm_caption_stage.psutil,
        "process_iter",
        lambda _attrs: [orphan, live_sibling, unrelated_orphan],
    )
    monkeypatch.setattr(vllm_caption_stage.psutil, "wait_procs", lambda procs, **_kwargs: (procs, []))

    stage._kill_orphan_engine_cores()

    orphan.kill.assert_called_once()
    live_sibling.kill.assert_not_called()
    unrelated_orphan.kill.assert_not_called()


def test_watchdog_start_is_idempotent_and_stops_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second start while alive reuses the running thread; stop joins and clears it."""
    stage = _make_watchdog_stage()
    stage._llm = object()  # keep the loop from early-returning on ``_llm is None``
    monkeypatch.setattr(vllm_caption_stage, "_ENGINE_CORE_WATCHDOG_INTERVAL_S", 0.01)
    monkeypatch.setattr(stage, "_engine_core_is_alive", lambda: True)

    stage._start_engine_core_watchdog()
    first_thread = stage._watchdog_thread
    assert first_thread is not None
    assert first_thread.is_alive()

    stage._start_engine_core_watchdog()
    assert stage._watchdog_thread is first_thread  # no second thread spawned

    stage._stop_engine_core_watchdog()
    assert stage._watchdog_thread is None
    first_thread.join(timeout=5.0)
    assert not first_thread.is_alive()


def test_watchdog_is_not_resurrected_after_stop_then_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the watchdog resurrection bug.

    If ``_stop_engine_core_watchdog`` cannot join the thread within its deadline (the
    thread is wedged mid liveness check), it nulls ``_watchdog_thread`` while the old
    thread is still alive. The previous implementation reused and ``clear()``-ed the
    shared stop ``Event`` on the next start, which un-signalled the stale thread's
    ``wait()`` and resurrected it -- leaving two live watchdogs. The fix installs a
    fresh ``Event`` per start (plus an identity guard in the loop), so the stale thread
    observes its own already-``set()`` event and exits.
    """
    stage = _make_watchdog_stage()
    stage._llm = object()
    monkeypatch.setattr(vllm_caption_stage, "_ENGINE_CORE_WATCHDOG_INTERVAL_S", 0.01)

    call_started = threading.Event()
    release_gate = threading.Event()

    def gated_alive() -> bool:
        # Mark that a watchdog iteration reached the liveness check, then block until
        # the test releases it. Returning True prevents the loop from calling os._exit.
        call_started.set()
        release_gate.wait(timeout=10.0)
        return True

    monkeypatch.setattr(stage, "_engine_core_is_alive", gated_alive)

    # Start the first watchdog and wait until it is wedged inside the gated check.
    stage._start_engine_core_watchdog()
    old_thread = stage._watchdog_thread
    assert old_thread is not None
    assert call_started.wait(timeout=5.0)

    # Stop it: the thread is blocked in the gated check, so join() times out and
    # ``_watchdog_thread`` is reset to None while the old thread is still alive.
    stage._stop_engine_core_watchdog()
    assert old_thread.is_alive()

    # Start a replacement. With the fix this installs a brand-new Event.
    stage._start_engine_core_watchdog()
    new_thread = stage._watchdog_thread
    assert new_thread is not None
    assert new_thread is not old_thread

    # Release the gate: the stale thread must observe its own already-set() event
    # (and the identity guard) and exit instead of looping forever.
    release_gate.set()
    old_thread.join(timeout=5.0)
    assert not old_thread.is_alive(), "stale watchdog thread was resurrected"
    assert new_thread.is_alive()

    # Cleanup the replacement watchdog.
    stage._stop_engine_core_watchdog()
    new_thread.join(timeout=5.0)
    assert not new_thread.is_alive()
