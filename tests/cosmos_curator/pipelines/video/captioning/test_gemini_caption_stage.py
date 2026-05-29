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

"""Tests for the Gemini-backed API caption stage."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini
from cosmos_curator.pipelines.common.api_caption_utils import normalize_gemini_response
from cosmos_curator.pipelines.video.captioning import gemini_caption_stage
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage, GeminiCaptionStage
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    Clip,
    SplitPipeTask,
    Video,
    Window,
    WindowConfig,
)


class _DummyModels:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        return SimpleNamespace(text="dummy caption")


class _DummyClient:
    def __init__(self, models: _DummyModels) -> None:
        self.models = models


class _EmptyResponseModels(_DummyModels):
    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])


def _make_task(mp4_bytes: bytes | None) -> SplitPipeTask:
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0))
    clip.windows.append(Window(start_frame=0, end_frame=1, mp4_bytes=mp4_bytes))
    video = Video(input_video=Path("source.mp4"))
    video.clips.append(clip)
    return SplitPipeTask(session_id="test-session", video=video)


def _attach_gemini_async_client(stage: GeminiCaptionStage, generate_content: AsyncMock) -> None:
    stage._async_client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    stage._runner = asyncio.Runner()


def _run_caption(stage: GeminiCaptionStage, window: Window) -> CaptionResult:
    stage._runner = asyncio.Runner()
    try:
        result, _detail = stage._runner.run(stage._generate_caption_with_error_detail_async(window))
        return result
    finally:
        stage.destroy()


@patch("cosmos_curator.pipelines.video.captioning.gemini_caption_stage.windowing_utils.make_windows_for_video")
def test_api_prep_stage_creates_windows_without_frames(mock_make_windows: MagicMock) -> None:
    """ApiPrepStage should avoid frame decoding while keeping MP4 bytes."""
    window_config = WindowConfig()
    stage = ApiPrepStage(window_config, num_cpus_for_prepare=2.0)
    mock_make_windows.return_value = ([Window(start_frame=0, end_frame=1)], [])
    video = Video(input_video=Path("test.mp4"))

    stage._prep_windows(video)

    mock_make_windows.assert_called_once()
    _, kwargs = mock_make_windows.call_args
    assert kwargs["return_frames"] is False
    assert kwargs["keep_mp4"] is True


def test_api_prep_stage_processes_each_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ApiPrepStage invokes window prep per task."""
    window_config = WindowConfig()
    stage = ApiPrepStage(window_config)
    spy = MagicMock()
    monkeypatch.setattr(stage, "_prep_windows", spy)
    task = _make_task(b"\x00\x01")
    task.stage_perf = {}

    stage.process_data([task, task])

    assert spy.call_count == 2


def test_gemini_caption_stage_generates_captions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Export captions into the window map when the API call succeeds."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    stage = GeminiCaptionStage(
        model_variant="gemini",
        model_name="models/test",
        max_output_tokens=64,
    )
    dummy_models = _DummyModels()
    _attach_gemini_async_client(
        stage,
        AsyncMock(side_effect=dummy_models.generate_content),
    )

    task = _make_task(b"\x00\x01")
    result = stage.process_data([task])

    window = result[0].video.clips[0].windows[0]
    assert window.caption["gemini"] == "dummy caption"
    assert window.caption_status == "success"
    assert window.caption_failure_reason is None
    assert dummy_models.last_kwargs is not None
    assert dummy_models.last_kwargs["model"] == "models/test"


def test_gemini_caption_stage_records_error_for_missing_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record an error when mp4 bytes are unavailable for a window."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    stage = GeminiCaptionStage()
    _attach_gemini_async_client(stage, AsyncMock(return_value=SimpleNamespace(text="ignored")))

    task = _make_task(None)
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    clip = task.video.clips[0]
    assert "gemini_caption_0" in clip.errors
    assert "gemini" not in clip.windows[0].caption
    assert clip.windows[0].caption_status == "error"
    assert clip.windows[0].caption_failure_reason == "exception"


def test_stage_setup_loads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure stage_setup pulls the API key from the config file."""

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.aio = SimpleNamespace(models=SimpleNamespace())

    monkeypatch.setattr(gemini_caption_stage, "genai", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(gemini_caption_stage, "genai_types", object())
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="config-key")),
    )

    stage = GeminiCaptionStage()
    stage.stage_setup()
    assert isinstance(stage._client, _FakeClient)
    assert stage._client.api_key == "config-key"  # type: ignore[unreachable]


def test_stage_setup_creates_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should create the aio client and runner."""

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.aio = SimpleNamespace(models=SimpleNamespace())

    monkeypatch.setattr(gemini_caption_stage, "genai", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(gemini_caption_stage, "genai_types", object())
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="config-key")),
    )

    stage = GeminiCaptionStage()
    stage.stage_setup()

    assert isinstance(stage._client, _FakeClient)
    assert stage._async_client is stage._client.aio  # type: ignore[union-attr]
    assert stage._runner is not None
    stage.destroy()


def test_gemini_caption_stage_logs_error_on_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture Gemini runtime errors when the response lacks text."""

    class _FakeClient:
        def __init__(self) -> None:
            self.models = _EmptyResponseModels()

    monkeypatch.setattr(gemini_caption_stage, "genai", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )

    stage = GeminiCaptionStage(verbose=True)
    _attach_gemini_async_client(
        stage, AsyncMock(return_value=SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))]))
    )
    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()
    clip = task.video.clips[0]
    assert "gemini_caption_0" in clip.errors
    assert clip.errors["gemini_caption_0"] == "Gemini response did not contain caption text."


def test_gemini_caption_stage_async_processes_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async path should write captions back to the correct windows."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )

    async def _generate_content(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text="async caption")

    stage = GeminiCaptionStage(batch_size=2)
    _attach_gemini_async_client(stage, AsyncMock(side_effect=_generate_content))

    task = _make_task(b"\x00\x01")
    task.video.clips[0].windows.append(Window(start_frame=1, end_frame=2, mp4_bytes=b"\x00\x02"))
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    clip = task.video.clips[0]
    assert clip.windows[0].caption["gemini"] == "async caption"
    assert clip.windows[1].caption["gemini"] == "async caption"


def test_stage_setup_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing config entry raises a runtime error."""

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    monkeypatch.setattr(gemini_caption_stage, "genai", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(gemini_caption_stage, "genai_types", object())
    monkeypatch.setattr(gemini_caption_stage, "load_config", ConfigFileData)

    with pytest.raises(RuntimeError, match="Gemini API key missing"):
        GeminiCaptionStage()


def test_normalize_response_handles_block_reason() -> None:
    """Prompt block reasons map to Blocked."""
    response = SimpleNamespace(prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
    result = normalize_gemini_response(response)
    assert result.outcome == CaptionOutcome.BLOCKED


def test_normalize_response_reports_truncated_finish_reason() -> None:
    """MAX_TOKENS with text maps to Truncated."""
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="partial")]),
                finish_reason="MAX_TOKENS",
            )
        ]
    )
    result = normalize_gemini_response(response)
    assert result.outcome == CaptionOutcome.TRUNCATED
    assert result.text == "partial"


def test_normalize_response_treats_empty_max_tokens_as_error() -> None:
    """MAX_TOKENS without text maps to Error rather than Truncated."""
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[]),
                finish_reason="MAX_TOKENS",
            )
        ]
    )
    result = normalize_gemini_response(response)
    assert result.outcome == CaptionOutcome.ERROR
    assert result.failure_reason == "exception"


def test_normalize_response_treats_recitation_as_blocked() -> None:
    """RECITATION wins over unexpected text."""
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="unexpected")]),
                finish_reason="RECITATION",
            )
        ]
    )
    result = normalize_gemini_response(response)
    assert result.outcome == CaptionOutcome.BLOCKED


def test_gemini_caption_stage_does_not_retry_invalid_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid retry loops when the Gemini API rejects the key."""

    class _FakeClientError(Exception):
        pass

    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.captioning.gemini_caption_stage.genai.errors",
        SimpleNamespace(ClientError=_FakeClientError),
        raising=False,
    )

    stage = GeminiCaptionStage()
    _attach_gemini_async_client(
        stage,
        AsyncMock(side_effect=_FakeClientError("API Key not found. Please pass a valid API key.")),
    )

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    clip = task.video.clips[0]
    assert clip.errors["gemini_caption_0"] == (
        "Gemini rejected the API key (API_KEY_INVALID). Update the cosmos-curator config with a valid Gemini API key."
    )
    assert clip.windows[0].caption_failure_reason == "exception"


def test_gemini_caption_stage_writes_truncated_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAX_TOKENS responses with text should write truncated status and keep text."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )

    class _TruncatedModels:
        def generate_content(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[SimpleNamespace(text="partial caption")]),
                        finish_reason="MAX_TOKENS",
                    )
                ]
            )

    stage = GeminiCaptionStage()
    _attach_gemini_async_client(
        stage,
        AsyncMock(
            return_value=SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[SimpleNamespace(text="partial caption")]),
                        finish_reason="MAX_TOKENS",
                    )
                ]
            )
        ),
    )

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    window = task.video.clips[0].windows[0]
    assert window.caption["gemini"] == "partial caption"
    assert window.caption_status == "truncated"
    assert window.caption_failure_reason is None


def test_gemini_caption_stage_writes_blocked_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety-style Gemini responses should write blocked status without text."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )

    class _BlockedModels:
        def generate_content(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(prompt_feedback=SimpleNamespace(block_reason="SAFETY"))

    stage = GeminiCaptionStage()
    _attach_gemini_async_client(
        stage, AsyncMock(return_value=SimpleNamespace(prompt_feedback=SimpleNamespace(block_reason="SAFETY")))
    )

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    window = task.video.clips[0].windows[0]
    assert "gemini" not in window.caption
    assert window.caption_status == "blocked"
    assert window.caption_failure_reason is None


def test_generate_caption_timeout_maps_to_timeout_failure_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeadlineExceeded maps to Error(timeout) after retries exhaust."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )

    class _FakeDeadlineExceededError(Exception):
        """Stand-in for Gemini timeout exceptions."""

    class _TimeoutModels:
        def generate_content(self, **_kwargs: object) -> SimpleNamespace:
            msg = "slow"
            raise _FakeDeadlineExceededError(msg)

    monkeypatch.setattr(gemini_caption_stage, "DeadlineExceeded", _FakeDeadlineExceededError)

    stage = GeminiCaptionStage(max_caption_retries=1, retry_delay_seconds=0)
    _attach_gemini_async_client(stage, AsyncMock(side_effect=_FakeDeadlineExceededError("slow")))

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.ERROR
    assert result.failure_reason == "timeout"
