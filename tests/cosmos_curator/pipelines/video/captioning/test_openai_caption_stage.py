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

"""Tests for the OpenAI-compatible API caption stage."""

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cosmos_curator.core.utils.config.config import ConfigFileData, OpenAIConfig, OpenAIEndpointConfig
from cosmos_curator.pipelines.video.captioning import openai_caption_stage
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    Clip,
    SplitPipeTask,
    Video,
    Window,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(mp4_bytes: bytes | None, *, num_windows: int = 1) -> SplitPipeTask:
    """Create a minimal SplitPipeTask with one clip and the given windows."""
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0))
    for i in range(num_windows):
        clip.windows.append(Window(start_frame=i * 10, end_frame=(i + 1) * 10, mp4_bytes=mp4_bytes))
    video = Video(input_video=Path("source.mp4"))
    video.clips.append(clip)
    return SplitPipeTask(session_id="test-session", video=video)


def _make_stage(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> OpenAICaptionStage:
    """Create a stage with the openai module patched so import-guarded code works."""
    # Ensure openai is importable in test (it may not live in "unified" env).
    monkeypatch.setattr(openai_caption_stage, "openai", _fake_openai_module(), raising=False)
    defaults: dict[str, object] = {
        "model_name": "test-model",
        "max_caption_retries": 1,
        "retry_delay_seconds": 0,
    }
    defaults.update(kwargs)
    return OpenAICaptionStage(**defaults)  # type: ignore[arg-type]


class _FakeChoice:
    """Minimal stand-in for openai ChatCompletionChoice."""

    def __init__(self, text: str | None, finish_reason: str = "stop") -> None:
        self.message = SimpleNamespace(content=text)
        self.finish_reason = finish_reason


class _FakeResponse:
    """Minimal stand-in for openai ChatCompletion."""

    def __init__(self, choices: list[_FakeChoice]) -> None:
        self.choices = choices


def _fake_openai_module() -> SimpleNamespace:
    """Return a fake openai module namespace with the error types the stage references."""

    class _AuthError(Exception):
        pass

    class _NotFoundError(Exception):
        pass

    class _BadRequestError(Exception):
        pass

    class _APITimeoutError(Exception):
        pass

    return SimpleNamespace(
        OpenAI=MagicMock,
        AsyncOpenAI=MagicMock,
        AuthenticationError=_AuthError,
        NotFoundError=_NotFoundError,
        BadRequestError=_BadRequestError,
        APITimeoutError=_APITimeoutError,
    )


def _attach_openai_async_client(stage: OpenAICaptionStage, create: AsyncMock) -> None:
    stage._async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    stage._runner = asyncio.Runner()


def _run_caption(stage: OpenAICaptionStage, window: Window) -> CaptionResult:
    stage._runner = asyncio.Runner()
    try:
        result, _detail = stage._runner.run(stage._generate_caption_with_error_detail_async(window))
        return result
    finally:
        stage.destroy()


# ---------------------------------------------------------------------------
# stage_setup
# ---------------------------------------------------------------------------


def test_stage_setup_creates_client_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should create an OpenAI client using api_key from config."""
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_openai = _fake_openai_module()
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)
    monkeypatch.setattr(
        openai_caption_stage,
        "maybe_load_config",
        lambda: ConfigFileData(openai=OpenAIConfig(caption=OpenAIEndpointConfig(api_key="test-key"))),
    )

    stage = OpenAICaptionStage(model_name="m")
    stage.stage_setup()

    assert captured["api_key"] == "test-key"
    assert "base_url" not in captured


def test_stage_setup_passes_base_url_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should forward base_url when present in config."""
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_openai = _fake_openai_module()
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)
    monkeypatch.setattr(
        openai_caption_stage,
        "maybe_load_config",
        lambda: ConfigFileData(
            openai=OpenAIConfig(caption=OpenAIEndpointConfig(api_key="k", base_url="http://localhost:8000/v1"))
        ),
    )

    stage = OpenAICaptionStage(model_name="m")
    stage.stage_setup()

    assert captured["base_url"] == "http://localhost:8000/v1"


def test_stage_setup_creates_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should create AsyncOpenAI and a runner."""
    sync_captured: dict[str, object] = {}
    async_captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            sync_captured.update(kwargs)

    class _FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            async_captured.update(kwargs)

    fake_openai = _fake_openai_module()
    fake_openai.OpenAI = _FakeClient
    fake_openai.AsyncOpenAI = _FakeAsyncClient
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)
    monkeypatch.setattr(
        openai_caption_stage,
        "maybe_load_config",
        lambda: ConfigFileData(
            openai=OpenAIConfig(caption=OpenAIEndpointConfig(api_key="k", base_url="http://localhost:8000/v1"))
        ),
    )

    stage = OpenAICaptionStage(model_name="m")
    stage.stage_setup()

    assert sync_captured["api_key"] == "k"
    assert async_captured["api_key"] == "k"
    assert async_captured["base_url"] == "http://localhost:8000/v1"
    assert stage._runner is not None
    stage.destroy()


def test_stage_setup_raises_when_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should raise RuntimeError when no config is loaded."""
    monkeypatch.setattr(openai_caption_stage, "openai", _fake_openai_module(), raising=False)
    monkeypatch.setattr(openai_caption_stage, "maybe_load_config", lambda: None)

    stage = OpenAICaptionStage(model_name="m")
    with pytest.raises(RuntimeError, match="OpenAI caption configuration not found"):
        stage.stage_setup()


def test_stage_setup_raises_when_openai_section_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should raise RuntimeError when openai section is absent."""
    monkeypatch.setattr(openai_caption_stage, "openai", _fake_openai_module(), raising=False)
    monkeypatch.setattr(openai_caption_stage, "maybe_load_config", ConfigFileData)

    stage = OpenAICaptionStage(model_name="m")
    with pytest.raises(RuntimeError, match="OpenAI caption configuration not found"):
        stage.stage_setup()


# ---------------------------------------------------------------------------
# _generate_caption_with_error_detail_async
# ---------------------------------------------------------------------------


def test_generate_caption_encodes_mp4_as_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    """The video payload should be base64-encoded with the correct data URI prefix."""
    stage = _make_stage(monkeypatch)

    raw_bytes = b"\x00\x01\x02\x03"
    expected_b64 = base64.b64encode(raw_bytes).decode("utf-8")

    create = AsyncMock(return_value=_FakeResponse([_FakeChoice("caption")]))
    _attach_openai_async_client(stage, create)

    stage.process_data([_make_task(raw_bytes)])

    call_kwargs = create.call_args.kwargs
    content_parts = call_kwargs["messages"][0]["content"]
    video_url = content_parts[0]["video_url"]["url"]
    assert video_url == f"data:video/mp4;base64,{expected_b64}"


def test_generate_caption_returns_stripped_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption should be stripped of leading/trailing whitespace."""
    stage = _make_stage(monkeypatch)

    create = AsyncMock(return_value=_FakeResponse([_FakeChoice("  hello world  ")]))
    _attach_openai_async_client(stage, create)

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.SUCCESS
    assert result.text == "hello world"


def test_generate_caption_returns_error_on_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty choices array maps to Error."""
    stage = _make_stage(monkeypatch)

    create = AsyncMock(return_value=_FakeResponse([]))
    _attach_openai_async_client(stage, create)

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.ERROR
    assert result.failure_reason == "exception"


def test_generate_caption_returns_error_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only content maps to Error."""
    stage = _make_stage(monkeypatch)

    create = AsyncMock(return_value=_FakeResponse([_FakeChoice("   ")]))
    _attach_openai_async_client(stage, create)

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.ERROR
    assert result.failure_reason == "exception"


def test_generate_caption_returns_truncated_on_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """A length finish reason with text maps to Truncated."""
    stage = _make_stage(monkeypatch)

    create = AsyncMock(return_value=_FakeResponse([_FakeChoice("partial", finish_reason="length")]))
    _attach_openai_async_client(stage, create)

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.TRUNCATED
    assert result.text == "partial"


def test_generate_caption_returns_blocked_on_content_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A content-filter response maps to Blocked even if text is unexpectedly present."""
    stage = _make_stage(monkeypatch)

    create = AsyncMock(return_value=_FakeResponse([_FakeChoice("unexpected partial", finish_reason="content_filter")]))
    _attach_openai_async_client(stage, create)

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.BLOCKED


def test_generate_caption_treats_null_content_without_content_filter_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Null content without an explicit content-filter finish reason maps to Error."""
    stage = _make_stage(monkeypatch)

    create = AsyncMock(return_value=_FakeResponse([_FakeChoice(None, finish_reason="stop")]))
    _attach_openai_async_client(stage, create)

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.ERROR
    assert result.failure_reason == "exception"


def test_generate_caption_raises_when_client_not_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError should be raised when stage_setup was not called."""
    stage = _make_stage(monkeypatch)
    with pytest.raises(RuntimeError, match="not initialized"):
        _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))


def test_generate_caption_returns_error_when_mp4_bytes_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing MP4 maps to Error."""
    stage = _make_stage(monkeypatch)
    _attach_openai_async_client(stage, AsyncMock(return_value=_FakeResponse([_FakeChoice("ignored")])))

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=None))
    assert result.outcome == CaptionOutcome.ERROR
    assert result.failure_reason == "exception"


# ---------------------------------------------------------------------------
# _process_task / process_data
# ---------------------------------------------------------------------------


def test_process_task_stores_caption_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful caption should be stored in window.caption[model_variant]."""
    stage = _make_stage(monkeypatch)

    _attach_openai_async_client(stage, AsyncMock(return_value=_FakeResponse([_FakeChoice("A nice caption")])))

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    window = task.video.clips[0].windows[0]
    assert window.caption["openai"] == "A nice caption"
    assert window.caption_status == "success"
    assert window.caption_failure_reason is None


def test_process_task_records_error_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When captioning fails, the error should be stored in clip.errors."""
    stage = _make_stage(monkeypatch)

    _attach_openai_async_client(stage, AsyncMock(side_effect=RuntimeError("API down")))

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    clip = task.video.clips[0]
    assert "openai_caption_0" in clip.errors
    assert clip.errors["openai_caption_0"] == "API down"
    assert "openai" not in clip.windows[0].caption
    assert clip.windows[0].caption_status == "error"
    assert clip.windows[0].caption_failure_reason == "exception"


def test_process_task_continues_after_window_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure in one window should not prevent subsequent windows from being captioned."""
    stage = _make_stage(monkeypatch)

    _attach_openai_async_client(
        stage, AsyncMock(side_effect=[RuntimeError("fail"), _FakeResponse([_FakeChoice("ok")])])
    )

    task = _make_task(b"\x00\x01", num_windows=2)
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    clip = task.video.clips[0]
    assert "openai_caption_0" in clip.errors
    assert clip.windows[1].caption["openai"] == "ok"
    assert clip.windows[0].caption_status == "error"
    assert clip.windows[1].caption_status == "success"


def test_process_task_writes_blocked_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocked responses should write caption_status without a caption payload."""
    stage = _make_stage(monkeypatch)

    _attach_openai_async_client(
        stage, AsyncMock(return_value=_FakeResponse([_FakeChoice(None, finish_reason="content_filter")]))
    )

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    window = task.video.clips[0].windows[0]
    assert "openai" not in window.caption
    assert window.caption_status == "blocked"
    assert window.caption_failure_reason is None


def test_process_task_writes_truncated_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncated responses should keep text and write truncated status."""
    stage = _make_stage(monkeypatch)

    _attach_openai_async_client(
        stage, AsyncMock(return_value=_FakeResponse([_FakeChoice("partial", finish_reason="length")]))
    )

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    window = task.video.clips[0].windows[0]
    assert window.caption["openai"] == "partial"
    assert window.caption_status == "truncated"
    assert window.caption_failure_reason is None


def test_process_task_records_detail_for_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handled API errors should keep a descriptive message in clip.errors."""
    stage = _make_stage(monkeypatch)

    _attach_openai_async_client(stage, AsyncMock(return_value=_FakeResponse([])))

    task = _make_task(b"\x00\x01")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    clip = task.video.clips[0]
    assert clip.errors["openai_caption_0"] == "OpenAI-compatible API returned no choices."
    assert clip.windows[0].caption_status == "error"
    assert clip.windows[0].caption_failure_reason == "exception"


def test_process_data_returns_all_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """process_data should return every input task without dropping any."""
    stage = _make_stage(monkeypatch)
    _attach_openai_async_client(stage, AsyncMock(return_value=_FakeResponse([_FakeChoice("c")])))

    tasks = [_make_task(b"\x00") for _ in range(3)]
    result = stage.process_data(tasks)

    assert len(result) == 3
    for orig, returned in zip(tasks, result, strict=False):
        assert orig is returned


def test_process_data_async_captions_each_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async path should write captions back to the correct windows."""
    stage = _make_stage(monkeypatch, batch_size=2)

    async def _create(**_: object) -> _FakeResponse:
        return _FakeResponse([_FakeChoice("async caption")])

    _attach_openai_async_client(stage, AsyncMock(side_effect=_create))

    task = _make_task(b"\x00\x01", num_windows=2)
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    clip = task.video.clips[0]
    assert clip.windows[0].caption["openai"] == "async caption"
    assert clip.windows[1].caption["openai"] == "async caption"


def test_process_data_async_interleaving_preserves_window_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-order async completion and one failure should still update the correct windows."""
    stage = _make_stage(monkeypatch, batch_size=3)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)

    payload_to_outcome = {
        base64.b64encode(b"\x00").decode("utf-8"): ("first", 0.03, None),
        base64.b64encode(b"\x01").decode("utf-8"): ("second", 0.0, None),
        base64.b64encode(b"\x02").decode("utf-8"): (None, 0.01, RuntimeError("boom")),
    }

    async def _create(**kwargs: object) -> _FakeResponse:
        messages = kwargs["messages"]  # type: ignore[index]
        content_parts = messages[0]["content"]  # type: ignore[index]
        url = content_parts[0]["video_url"]["url"]  # type: ignore[index]
        payload = url.split(",", 1)[1]
        text, delay, error = payload_to_outcome[payload]
        await asyncio.sleep(delay)
        if error is not None:
            raise error
        return _FakeResponse([_FakeChoice(text)])

    _attach_openai_async_client(stage, AsyncMock(side_effect=_create))

    task = _make_task(b"\x00", num_windows=0)
    clip = task.video.clips[0]
    clip.windows.extend(
        [
            Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"),
            Window(start_frame=1, end_frame=2, mp4_bytes=b"\x01"),
            Window(start_frame=2, end_frame=3, mp4_bytes=b"\x02"),
        ]
    )

    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    assert clip.windows[0].caption["openai"] == "first"
    assert clip.windows[1].caption["openai"] == "second"
    assert clip.windows[2].caption_status == "error"
    assert clip.errors["openai_caption_2"] == "boom"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_generate_caption_does_not_retry_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """AuthenticationError should not be retried."""
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)

    stage = OpenAICaptionStage(model_name="m", max_caption_retries=3, retry_delay_seconds=0)
    create = AsyncMock(side_effect=fake_openai.AuthenticationError("bad key"))
    _attach_openai_async_client(stage, create)

    task = _make_task(b"\x00")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    assert create.call_count == 1
    assert "openai_caption_0" in task.video.clips[0].errors
    assert task.video.clips[0].errors["openai_caption_0"] == "bad key"
    assert task.video.clips[0].windows[0].caption_failure_reason == "exception"


def test_generate_caption_does_not_retry_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """BadRequestError should not be retried."""
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)

    stage = OpenAICaptionStage(model_name="m", max_caption_retries=3, retry_delay_seconds=0)
    create = AsyncMock(side_effect=fake_openai.BadRequestError("invalid"))
    _attach_openai_async_client(stage, create)

    task = _make_task(b"\x00")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    assert create.call_count == 1
    assert task.video.clips[0].errors["openai_caption_0"] == "invalid"
    assert task.video.clips[0].windows[0].caption_failure_reason == "exception"


def test_generate_caption_retries_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient errors should be retried up to max_caption_retries times."""
    stage = _make_stage(monkeypatch, max_caption_retries=3, retry_delay_seconds=0)

    create = AsyncMock(
        side_effect=[
            ConnectionError("transient"),
            ConnectionError("transient"),
            _FakeResponse([_FakeChoice("recovered")]),
        ]
    )
    _attach_openai_async_client(stage, create)

    task = _make_task(b"\x00")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    assert create.call_count == 3
    assert task.video.clips[0].windows[0].caption["openai"] == "recovered"
    assert task.video.clips[0].windows[0].caption_status == "success"


def test_generate_caption_timeout_maps_to_timeout_failure_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhausted API timeouts map to Error(timeout)."""
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)

    stage = OpenAICaptionStage(model_name="m", max_caption_retries=1, retry_delay_seconds=0)
    _attach_openai_async_client(stage, AsyncMock(side_effect=fake_openai.APITimeoutError("slow")))

    result = _run_caption(stage, Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result.outcome == CaptionOutcome.ERROR
    assert result.failure_reason == "timeout"
