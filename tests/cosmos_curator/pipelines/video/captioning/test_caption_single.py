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

"""Tests for ``caption_single`` on each per-window caption stage.

Exercises the new :class:`SingleInferenceCaptionStage`-shaped entry
point so ``PerEventCaptionStage`` can rely on a uniform
``caption_single(prompt, video_bytes) -> str`` contract across all four
backends. Heavy backends (``VllmCaptionStage``, ``VllmAsyncCaptionStage``)
are gated with the ``default`` env marker; the CPU-only OpenAI / Gemini
paths run on every test invocation.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cosmos_curator.core.utils.config.config import (
    ConfigFileData,
    Gemini,
)
from cosmos_curator.pipelines.video.captioning import gemini_caption_stage, openai_caption_stage
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import (
    GeminiCaptionStage,
    GeminiRetryPolicy,
)
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.captioning.single_inference import SingleInferenceCaptionStage

# ---------------------------------------------------------------------------
# ABC membership — quick sanity check
# ---------------------------------------------------------------------------


def test_openai_stage_implements_single_inference_protocol() -> None:
    """``OpenAICaptionStage`` should subclass ``SingleInferenceCaptionStage``."""
    assert issubclass(OpenAICaptionStage, SingleInferenceCaptionStage)
    stage = OpenAICaptionStage(model_name="m")
    assert isinstance(stage, SingleInferenceCaptionStage)


def test_gemini_stage_implements_single_inference_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GeminiCaptionStage`` should subclass ``SingleInferenceCaptionStage``."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="key-xyz")),
    )
    assert issubclass(GeminiCaptionStage, SingleInferenceCaptionStage)
    stage = GeminiCaptionStage()
    assert isinstance(stage, SingleInferenceCaptionStage)


# ---------------------------------------------------------------------------
# OpenAICaptionStage.caption_single
# ---------------------------------------------------------------------------


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
    stage._async_client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    stage._runner = asyncio.Runner()


def _make_openai_stage(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> OpenAICaptionStage:
    monkeypatch.setattr(openai_caption_stage, "openai", _fake_openai_module(), raising=False)
    defaults: dict[str, object] = {
        "model_name": "test-model",
        "max_caption_retries": 1,
        "retry_delay_seconds": 0,
    }
    defaults.update(kwargs)
    return OpenAICaptionStage(**defaults)  # type: ignore[arg-type]


def test_openai_caption_single_returns_text_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: a populated chat-completion response yields the caption text."""
    stage = _make_openai_stage(monkeypatch)
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"), finish_reason="stop")],
        ),
    )
    _attach_openai_async_client(stage, create)
    try:
        result = stage.caption_single("describe this", b"\x00fake-mp4")
    finally:
        stage.destroy()
    assert result == "hello world"
    # Each request encodes the video bytes once and forwards the prompt verbatim.
    create.assert_awaited_once()
    awaited_kwargs = create.await_args.kwargs
    content = awaited_kwargs["messages"][0]["content"]
    assert any(part.get("type") == "video_url" for part in content)
    assert any(part.get("text") == "describe this" for part in content)


def test_openai_caption_single_raises_when_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response flagged ``BLOCKED`` propagates as ``RuntimeError``."""
    stage = _make_openai_stage(monkeypatch)
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="content_filter")],
        ),
    )
    _attach_openai_async_client(stage, create)
    try:
        with pytest.raises(RuntimeError, match="blocked"):
            stage.caption_single("p", b"x")
    finally:
        stage.destroy()


def test_openai_caption_single_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ERROR result with no caption text raises ``RuntimeError``.

    ``BadRequestError`` is short-circuited by the retry policy and surfaces
    via :func:`openai_error_result_from_exception`, so ``caption_single``
    re-raises the provider's message verbatim (``"nope"`` here).
    """
    stage = _make_openai_stage(monkeypatch)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)
    create = AsyncMock(side_effect=fake_openai.BadRequestError("nope"))
    _attach_openai_async_client(stage, create)
    try:
        with pytest.raises(RuntimeError, match="nope"):
            stage.caption_single("p", b"x")
    finally:
        stage.destroy()


def test_openai_caption_single_raises_without_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling without ``stage_setup`` (no runner / async client) raises."""
    stage = _make_openai_stage(monkeypatch)
    with pytest.raises(RuntimeError, match="runner not initialized"):
        stage.caption_single("p", b"x")


# ---------------------------------------------------------------------------
# GeminiCaptionStage.caption_single
# ---------------------------------------------------------------------------


def _patch_gemini_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="key-xyz")),
    )


def test_gemini_caption_single_returns_text_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: ``client.models.generate_content`` text is returned verbatim."""
    _patch_gemini_config(monkeypatch)
    stage = GeminiCaptionStage(
        max_caption_retries=1,
        caption_single_options=gemini_caption_stage.CaptionSingleOptions(retry_policy=GeminiRetryPolicy.FIXED),
    )
    captured: dict[str, object] = {}

    def _gen(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(text="caption-text", candidates=[], usage_metadata=None)

    stage._client = SimpleNamespace(models=SimpleNamespace(generate_content=_gen))  # type: ignore[assignment]
    out = stage.caption_single("describe", b"\x00mp4")
    assert out == "caption-text"
    # Diagnostic accessors are populated even on success.
    assert stage.last_finish_reasons == []


def test_gemini_caption_single_uses_exponential_jitter_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EXPONENTIAL_JITTER`` policy retries transient errors."""
    _patch_gemini_config(monkeypatch)
    stage = GeminiCaptionStage(
        max_caption_retries=3,
        retry_delay_seconds=0.0,
        caption_single_options=gemini_caption_stage.CaptionSingleOptions(
            retry_policy=GeminiRetryPolicy.EXPONENTIAL_JITTER,
            retry_max_delay_seconds=0.0,
            retry_jitter_seconds=0.0,
        ),
    )

    transient = SimpleNamespace(code=503)
    err = type("APIError", (Exception,), {})  # name-based transient classification
    raised = err("server overloaded")
    setattr(raised, "code", 503)  # noqa: B010

    call_count = {"n": 0}

    def _gen(**_kwargs: object) -> SimpleNamespace:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise raised
        return SimpleNamespace(text="ok", candidates=[], usage_metadata=None)

    stage._client = SimpleNamespace(models=SimpleNamespace(generate_content=_gen))  # type: ignore[assignment]
    out = stage.caption_single("p", b"x")
    assert out == "ok"
    assert call_count["n"] == 3
    # Silence unused-variable lint without changing behavior.
    assert transient.code == 503


def test_gemini_caption_single_raises_when_files_api_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``enable_files_api_fallback`` an oversize clip raises immediately."""
    _patch_gemini_config(monkeypatch)
    stage = GeminiCaptionStage(
        max_video_size_bytes=8,
        max_caption_retries=1,
        caption_single_options=gemini_caption_stage.CaptionSingleOptions(
            retry_policy=GeminiRetryPolicy.FIXED,
            enable_files_api_fallback=False,
        ),
    )
    stage._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kw: SimpleNamespace(text="x")))  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="exceeds Gemini inline limit"):
        stage.caption_single("p", b"x" * 64)


def test_gemini_caption_single_raises_without_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """``stage._client is None`` should fail fast with a clear message."""
    _patch_gemini_config(monkeypatch)
    stage = GeminiCaptionStage()
    with pytest.raises(RuntimeError, match="sync client not initialized"):
        stage.caption_single("p", b"x")


def test_gemini_caption_single_extracts_text_from_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to ``candidates[*].content.parts[*].text`` when ``response.text`` is empty."""
    _patch_gemini_config(monkeypatch)
    stage = GeminiCaptionStage(
        max_caption_retries=1,
        caption_single_options=gemini_caption_stage.CaptionSingleOptions(retry_policy=GeminiRetryPolicy.FIXED),
    )
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="from candidate")]),
                finish_reason="STOP",
            ),
        ],
        usage_metadata="some-usage",
    )
    stage._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kw: response))  # type: ignore[assignment]
    out = stage.caption_single("p", b"x")
    assert out == "from candidate"
    assert stage.last_finish_reasons == ["STOP"]
    assert stage.last_usage_metadata == "some-usage"


# ---------------------------------------------------------------------------
# VllmCaptionStage.caption_single — env=default
# ---------------------------------------------------------------------------


@pytest.mark.env("default")
def test_vllm_caption_single_returns_text_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """``caption_single`` decodes once, builds one llm_input, and returns engine text."""
    from cosmos_curator.pipelines.video.captioning import vllm_caption_stage as vcs  # noqa: PLC0415
    from cosmos_curator.pipelines.video.utils.data_model import VllmConfig  # noqa: PLC0415

    stage = vcs.VllmCaptionStage(vllm_config=VllmConfig(model_variant="qwen"))

    monkeypatch.setattr(
        vcs,
        "make_model_inputs",
        lambda **_kw: [{"prompt": "rendered", "multi_modal_data": {"video": [None]}}],
    )
    monkeypatch.setattr(
        vcs.VllmCaptionStage,
        "_decode_video_for_caption_single",
        lambda _self, _bytes: (None, {}),
    )

    fake_engine = MagicMock()
    fake_engine.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text="caption-from-engine", finish_reason="stop")]),
    ]
    stage._llm = fake_engine
    stage._processor = SimpleNamespace()
    stage._caption_single_sampling_params = SimpleNamespace(max_tokens=4096)

    out = stage.caption_single("describe", b"\x00mp4")
    assert out == "caption-from-engine"
    fake_engine.generate.assert_called_once()


@pytest.mark.env("default")
def test_vllm_caption_single_clears_video_max_pixels_without_mutating_stage_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``caption_single`` remains out of scope for sync window resize overrides."""
    from cosmos_curator.pipelines.video.captioning import vllm_caption_stage as vcs  # noqa: PLC0415
    from cosmos_curator.pipelines.video.utils.data_model import VllmConfig  # noqa: PLC0415

    stage = vcs.VllmCaptionStage(
        vllm_config=VllmConfig(model_variant="qwen3_vl_30b", video_max_pixels_per_frame=602112)
    )
    captured: dict[str, object] = {}

    def fake_make_model_inputs(**kwargs: object) -> list[dict[str, object]]:
        captured["config"] = kwargs["config"]
        return [{"prompt": "rendered", "multi_modal_data": {"video": [None]}}]

    monkeypatch.setattr(vcs, "make_model_inputs", fake_make_model_inputs)
    monkeypatch.setattr(
        vcs.VllmCaptionStage,
        "_decode_video_for_caption_single",
        lambda _self, _bytes: (None, {}),
    )

    fake_engine = MagicMock()
    fake_engine.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text="caption-from-engine", finish_reason="stop")]),
    ]
    stage._llm = fake_engine
    stage._processor = SimpleNamespace()
    stage._caption_single_sampling_params = SimpleNamespace(max_tokens=4096)

    out = stage.caption_single("describe", b"\x00mp4")

    assert out == "caption-from-engine"
    config = captured["config"]
    assert isinstance(config, VllmConfig)
    assert config.video_max_pixels_per_frame is None
    assert stage._vllm_config.video_max_pixels_per_frame == 602112


@pytest.mark.env("default")
def test_vllm_caption_single_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty engine output raises ``RuntimeError`` with finish_reason in the message."""
    from cosmos_curator.pipelines.video.captioning import vllm_caption_stage as vcs  # noqa: PLC0415
    from cosmos_curator.pipelines.video.utils.data_model import VllmConfig  # noqa: PLC0415

    stage = vcs.VllmCaptionStage(vllm_config=VllmConfig(model_variant="qwen"))
    monkeypatch.setattr(
        vcs, "make_model_inputs", lambda **_kw: [{"prompt": "x", "multi_modal_data": {"video": [None]}}]
    )
    monkeypatch.setattr(
        vcs.VllmCaptionStage,
        "_decode_video_for_caption_single",
        lambda _self, _bytes: (None, {}),
    )
    fake_engine = MagicMock()
    fake_engine.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text="", finish_reason="length")]),
    ]
    stage._llm = fake_engine
    stage._processor = SimpleNamespace()
    stage._caption_single_sampling_params = SimpleNamespace(max_tokens=4096)
    with pytest.raises(RuntimeError, match="empty caption"):
        stage.caption_single("p", b"x")


# ---------------------------------------------------------------------------
# VllmAsyncCaptionStage.caption_single — env=default
# ---------------------------------------------------------------------------


@pytest.mark.env("default")
def test_vllm_async_caption_single_returns_text_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """``caption_single`` builds one llm_input, drives ``engine.generate`` to completion."""
    from cosmos_curator.pipelines.video.captioning import vllm_async_stage as vas  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig  # noqa: PLC0415

    stage = vas.VllmAsyncCaptionStage(serve_config=VllmAsyncConfig(model_variant="qwen"), model_name="qwen")

    monkeypatch.setattr(
        vas.VllmAsyncCaptionStage,
        "_decode_video_for_caption_single",
        lambda _self, _bytes: (None, {}),
    )
    monkeypatch.setattr(
        vas,
        "make_model_inputs",
        lambda **_kw: [{"prompt": "rendered", "multi_modal_data": {"video": [None]}}],
    )

    async def _fake_generate(*_args: object, **_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(outputs=[SimpleNamespace(text="caption-from-async", finish_reason="stop")])

    fake_engine = MagicMock()
    fake_engine.generate = _fake_generate
    stage._engine = fake_engine
    stage._processor = SimpleNamespace()
    stage._caption_single_sampling_params = SimpleNamespace(max_tokens=4096)

    try:
        out = stage.caption_single("describe", b"\x00mp4")
        assert out == "caption-from-async"
    finally:
        if stage._caption_single_runner is not None:
            stage._caption_single_runner.close()
            stage._caption_single_runner = None


@pytest.mark.env("default")
def test_vllm_async_caption_single_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty engine output raises ``RuntimeError`` with finish_reason in the message."""
    from cosmos_curator.pipelines.video.captioning import vllm_async_stage as vas  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig  # noqa: PLC0415

    stage = vas.VllmAsyncCaptionStage(serve_config=VllmAsyncConfig(model_variant="qwen"), model_name="qwen")

    monkeypatch.setattr(
        vas.VllmAsyncCaptionStage,
        "_decode_video_for_caption_single",
        lambda _self, _bytes: (None, {}),
    )
    monkeypatch.setattr(
        vas,
        "make_model_inputs",
        lambda **_kw: [{"prompt": "rendered", "multi_modal_data": {"video": [None]}}],
    )

    async def _fake_generate(*_args: object, **_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(outputs=[SimpleNamespace(text="", finish_reason="length")])

    fake_engine = MagicMock()
    fake_engine.generate = _fake_generate
    stage._engine = fake_engine
    stage._processor = SimpleNamespace()
    stage._caption_single_sampling_params = SimpleNamespace(max_tokens=4096)

    try:
        with pytest.raises(RuntimeError, match="empty caption"):
            stage.caption_single("p", b"x")
    finally:
        if stage._caption_single_runner is not None:
            stage._caption_single_runner.close()
            stage._caption_single_runner = None


# ---------------------------------------------------------------------------
# _decode_video_for_caption_single — inclusive frame-bound regression
# ---------------------------------------------------------------------------
#
# ``WindowFrameInfo.end`` is documented as inclusive (windowing_types.py:22)
# and the canonical builder uses ``total_frames - 1``
# (windowing_utils.py:58). ``read_video_cpu`` / ``fetch_video`` both treat
# ``end`` as inclusive (vision_process.py:133 computes
# ``end - start + 1``). Passing ``end=total_frames`` would request a frame
# at out-of-range index ``total_frames``. These tests pin the inclusive
# upper bound so the off-by-one can't reappear.


@pytest.mark.env("default")
def test_vllm_decode_video_for_caption_single_uses_inclusive_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``WindowFrameInfo.end`` is inclusive — caption_single must pass ``total_frames - 1``."""
    from cosmos_curator.pipelines.video.captioning import vllm_caption_stage as vcs  # noqa: PLC0415
    from cosmos_curator.pipelines.video.utils.data_model import VllmConfig  # noqa: PLC0415

    captured: dict[str, object] = {}

    def _fake_fetch_video(
        _path: object,
        *,
        sampling_fps: float,  # noqa: ARG001
        window_range: object,
        **_kw: object,
    ) -> tuple[object, list[int]]:
        captured["window_range"] = window_range
        return SimpleNamespace(shape=(10,), ndim=1), [10]

    @contextlib.contextmanager
    def _fake_memfd(*_args: object, **_kw: object) -> Iterator[str]:
        yield "fake-mp4-path"

    monkeypatch.setattr(vcs, "get_frame_count", lambda _b: 100)
    monkeypatch.setattr(vcs, "fetch_video", _fake_fetch_video)
    monkeypatch.setattr(vcs, "buffer_as_memfd_path", _fake_memfd)

    stage = vcs.VllmCaptionStage(vllm_config=VllmConfig(model_variant="qwen"))
    stage._decode_video_for_caption_single(b"\x00mp4")

    window_range = captured["window_range"]
    assert isinstance(window_range, list)
    assert len(window_range) == 1
    window = window_range[0]
    assert window.start == 0
    assert window.end == 99  # total_frames=100, inclusive upper bound


@pytest.mark.env("default")
def test_vllm_decode_video_for_caption_single_qwen3_uses_inclusive_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model-side preprocessing takes the ``read_video_cpu`` branch — same inclusive contract."""
    from cosmos_curator.pipelines.common.model_constraints import PreprocessMode  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning import vllm_caption_stage as vcs  # noqa: PLC0415
    from cosmos_curator.pipelines.video.utils.data_model import VllmConfig  # noqa: PLC0415

    captured: dict[str, object] = {}

    def _fake_read_video_cpu(
        _path: object,
        _sampling_fps: float,
        _num_frames: int,
        window_range: object,
    ) -> tuple[object, list[int]]:
        captured["window_range"] = window_range
        return SimpleNamespace(shape=(8,), ndim=1), [8]

    @contextlib.contextmanager
    def _fake_memfd(*_args: object, **_kw: object) -> Iterator[str]:
        yield "fake-mp4-path"

    monkeypatch.setattr(vcs, "get_frame_count", lambda _b: 64)
    monkeypatch.setattr(vcs, "read_video_cpu", _fake_read_video_cpu)
    monkeypatch.setattr(vcs, "buffer_as_memfd_path", _fake_memfd)

    stage = vcs.VllmCaptionStage(
        vllm_config=VllmConfig(model_variant="qwen3_vl_30b", preprocess_mode=PreprocessMode.MODEL)
    )
    stage._decode_video_for_caption_single(b"\x00mp4")

    window_range = captured["window_range"]
    assert isinstance(window_range, list)
    assert window_range[0].start == 0
    assert window_range[0].end == 63  # total_frames=64, inclusive upper bound


@pytest.mark.env("default")
def test_vllm_async_decode_video_for_caption_single_uses_inclusive_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``VllmAsyncCaptionStage`` mirrors the sync inclusive-end contract."""
    from cosmos_curator.pipelines.video.captioning import vllm_async_stage as vas  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig  # noqa: PLC0415

    captured: dict[str, object] = {}

    def _fake_fetch_video(
        _path: object,
        *,
        sampling_fps: float,  # noqa: ARG001
        window_range: object,
        **_kw: object,
    ) -> tuple[object, list[int]]:
        captured["window_range"] = window_range
        return SimpleNamespace(shape=(10,), ndim=1), [10]

    @contextlib.contextmanager
    def _fake_memfd(*_args: object, **_kw: object) -> Iterator[str]:
        yield "fake-mp4-path"

    monkeypatch.setattr(vas, "get_frame_count", lambda _b: 100)
    monkeypatch.setattr(vas, "fetch_video", _fake_fetch_video)
    monkeypatch.setattr(vas, "buffer_as_memfd_path", _fake_memfd)

    stage = vas.VllmAsyncCaptionStage(serve_config=VllmAsyncConfig(model_variant="qwen"), model_name="qwen")
    stage._decode_video_for_caption_single(b"\x00mp4")

    window_range = captured["window_range"]
    assert isinstance(window_range, list)
    assert len(window_range) == 1
    window = window_range[0]
    assert window.start == 0
    assert window.end == 99  # total_frames=100, inclusive upper bound


@pytest.mark.env("default")
def test_vllm_async_decode_video_for_caption_single_qwen3_uses_inclusive_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async model-side preprocessing also hits ``read_video_cpu`` — same inclusive contract."""
    from cosmos_curator.pipelines.common.model_constraints import PreprocessMode  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning import vllm_async_stage as vas  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig  # noqa: PLC0415

    captured: dict[str, object] = {}

    def _fake_read_video_cpu(
        _path: object,
        _sampling_fps: float,
        _num_frames: int,
        window_range: object,
    ) -> tuple[object, list[int]]:
        captured["window_range"] = window_range
        return SimpleNamespace(shape=(8,), ndim=1), [8]

    @contextlib.contextmanager
    def _fake_memfd(*_args: object, **_kw: object) -> Iterator[str]:
        yield "fake-mp4-path"

    monkeypatch.setattr(vas, "get_frame_count", lambda _b: 64)
    monkeypatch.setattr(vas, "read_video_cpu", _fake_read_video_cpu)
    monkeypatch.setattr(vas, "buffer_as_memfd_path", _fake_memfd)

    stage = vas.VllmAsyncCaptionStage(
        serve_config=VllmAsyncConfig(model_variant="qwen3_vl_30b", preprocess_mode=PreprocessMode.MODEL),
        model_name="qwen3_vl_30b",
    )
    stage._decode_video_for_caption_single(b"\x00mp4")

    window_range = captured["window_range"]
    assert isinstance(window_range, list)
    assert window_range[0].start == 0
    assert window_range[0].end == 63  # total_frames=64, inclusive upper bound
