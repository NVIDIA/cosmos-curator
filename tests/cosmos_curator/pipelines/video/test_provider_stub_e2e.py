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
"""CPU end-to-end coverage for video filter/classifier provider arms.

The tests parse the production split CLI, assemble the complete pipeline, and
run only the assembled API prep/provider/postprocess chain with
``SequentialRunner``. This exercises real video window preparation and provider
request handling without starting Ray, using GPUs, or writing outputs.
"""

import argparse
import base64
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini, OpenAIConfig, OpenAIEndpointConfig
from cosmos_curator.pipelines.video.captioning import gemini_caption_stage, openai_caption_stage
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage, GeminiCaptionStage
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.filtering.aesthetics.semantic_filter_stages import (
    VllmFilteringStage,
    VllmVideoClassifierStage,
)
from cosmos_curator.pipelines.video.splitting_pipeline import _assemble_stages, add_split_command
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask
from tests.cosmos_curator.pipelines.provider_stub_utils import GeminiStub, OpenAIStubServer, require_provider_module
from tests.cosmos_curator.pipelines.stage_test_utils import stage_object

Provider = Literal["openai", "gemini"]
ProviderStub = OpenAIStubServer | GeminiStub
PostprocessStage = VllmFilteringStage | VllmVideoClassifierStage


@dataclass(frozen=True)
class VideoProviderE2EContext:
    """Shared fixtures for one assembled video provider chain."""

    monkeypatch: pytest.MonkeyPatch
    tmp_path: pathlib.Path
    task: SplitPipeTask
    sequential_runner: RunnerInterface


@dataclass(frozen=True)
class ClassifierCase:
    """Expected classifier behavior for one provider response."""

    response: str
    classification: list[str]
    should_filter: bool


@pytest.fixture
def video_provider_e2e_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
) -> VideoProviderE2EContext:
    """Bundle common fixtures so provider tests remain concise."""
    return VideoProviderE2EContext(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        task=sample_filtering_task,
        sequential_runner=sequential_runner,
    )


def _parse_split_args(tmp_path: pathlib.Path, options: Sequence[str]) -> argparse.Namespace:
    """Parse provider options through the production ``split`` subcommand."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_split_command(subparsers)
    return parser.parse_args(
        [
            "split",
            "--input-video-path",
            str(tmp_path / "input"),
            "--output-clip-path",
            str(tmp_path / "output"),
            "--no-generate-captions",
            "--no-generate-embeddings",
            "--motion-filter",
            "disable",
            *options,
        ]
    )


def _extract_provider_chain(
    args: argparse.Namespace,
    postprocess_type: type[PostprocessStage],
    provider: Provider,
) -> list[CuratorStageSpec]:
    """Extract and validate the assembled API prep/provider/postprocess chain."""
    assembled = _assemble_stages(args)
    stages = [stage_object(stage) for stage in assembled]
    postprocess_index = next(index for index, stage in enumerate(stages) if isinstance(stage, postprocess_type))
    chain = stages[postprocess_index - 2 : postprocess_index + 1]

    assert len(chain) == 3
    assert isinstance(chain[0], ApiPrepStage)
    expected_provider_type = OpenAICaptionStage if provider == "openai" else GeminiCaptionStage
    assert isinstance(chain[1], expected_provider_type)
    assert isinstance(chain[2], postprocess_type)
    return [CuratorStageSpec(stage) for stage in chain]


def _configure_provider(
    provider: Provider,
    endpoint_key: Literal["filter", "classifier"],
    response_text: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> ProviderStub:
    """Point the production provider stage at its reusable test stub."""
    if provider == "openai":
        openai = require_provider_module("openai")
        server = cast("OpenAIStubServer", request.getfixturevalue("openai_stub_server"))
        endpoint = OpenAIEndpointConfig(api_key="test-key", base_url=server.base_url)
        openai_config = OpenAIConfig(filter=endpoint) if endpoint_key == "filter" else OpenAIConfig(classifier=endpoint)
        monkeypatch.setattr(openai_caption_stage, "openai", openai, raising=False)
        monkeypatch.setattr(
            openai_caption_stage,
            "maybe_load_config",
            lambda: ConfigFileData(openai=openai_config),
        )
        server.set_chat_content(response_text)
        return server

    stub_factory = cast("Callable[..., GeminiStub]", request.getfixturevalue("gemini_stub"))
    stub = stub_factory(text=response_text)
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="test-key")),
    )
    monkeypatch.setattr(
        gemini_caption_stage,
        "genai",
        SimpleNamespace(Client=lambda **_kwargs: stub.client),
    )
    return stub


def _assert_provider_request(
    stub: ProviderStub,
    provider: Provider,
    expected_model: str,
    prompt_marker: str,
    expected_max_tokens: int,
) -> None:
    """Assert the provider received one video request with the assembled config."""
    if provider == "openai":
        assert isinstance(stub, OpenAIStubServer)
        requests = stub.requests_for("/chat/completions")
        assert len(requests) == 1
        body = requests[0].body
        assert body is not None
        assert body["model"] == expected_model
        assert body["max_tokens"] == expected_max_tokens
        content = body["messages"][0]["content"]
        video_parts = [part for part in content if part["type"] == "video_url"]
        assert len(video_parts) == 1
        data_url = video_parts[0]["video_url"]["url"]
        prefix, encoded_video = data_url.split(",", maxsplit=1)
        assert prefix == "data:video/mp4;base64"
        assert base64.b64decode(encoded_video)
        prompt = "\n".join(part["text"] for part in content if part["type"] == "text")
        assert prompt_marker in prompt
        return

    assert isinstance(stub, GeminiStub)
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["model"] == expected_model
    assert call["config"].max_output_tokens == expected_max_tokens
    # The stage builds a real ``genai_types.Content`` and only ``genai`` is patched, so the
    # recorded contents are always SDK objects rather than plain dicts.
    parts = call["contents"].parts
    assert isinstance(parts, list)
    inline_parts = [part for part in parts if part.inline_data is not None]
    assert len(inline_parts) == 1
    inline_data = inline_parts[0].inline_data
    assert inline_data.mime_type == "video/mp4"
    assert inline_data.data
    text_parts = [part.text for part in parts if part.text is not None]
    assert any(prompt_marker in text for text in text_parts)


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_vlm_filter_score_only_provider_chain_from_split_cli(
    provider: Provider,
    request: pytest.FixtureRequest,
    video_provider_e2e_context: VideoProviderE2EContext,
) -> None:
    """Score-only filtering should retain the clip while recording provider results."""
    context = video_provider_e2e_context
    response = '{"slideshow": "yes"}'
    stub = _configure_provider(provider, "filter", response, request, context.monkeypatch)
    expected_model = "stub-filter-model" if provider == "openai" else "models/stub-filter"
    max_tokens = 123
    args = _parse_split_args(
        context.tmp_path,
        [
            "--vlm-filter",
            "score-only",
            "--vlm-filter-endpoint",
            provider,
            "--vlm-filter-categories",
            "slideshow",
            "--vlm-filter-openai-model-name",
            "stub-filter-model",
            "--vlm-filter-openai-retries",
            "1",
            "--vlm-filter-openai-retry-delay-seconds",
            "0",
            "--vlm-filter-gemini-model-name",
            "models/stub-filter",
            "--vlm-filter-gemini-retries",
            "1",
            "--vlm-filter-gemini-retry-delay-seconds",
            "0",
            "--vlm-filter-max-output-tokens",
            str(max_tokens),
        ],
    )
    chain = _extract_provider_chain(args, VllmFilteringStage, provider)

    result = context.sequential_runner.run([context.task], chain, "")

    assert result is not None
    assert len(result) == 1
    assert result[0] is context.task
    assert len(context.task.video.clips) == 1
    assert not context.task.video.filtered_clips
    assert context.task.video.clip_stats.num_filtered_by_qwen_semantic == 0
    clip = context.task.video.clips[0]
    assert clip.qwen_rejection_stage is None
    assert not clip.errors
    assert clip.extracted_frames.resolve() is not None
    assert len(clip.filter_windows) == 1
    window = clip.filter_windows[0]
    assert window.caption[provider] == response
    assert window.caption_status == "success"
    assert window.caption_failure_reason is None
    assert not window.errors
    assert window.mp4_bytes.resolve() is None
    assert "slideshow" in window.caption["qwen_rejection_reasons"]
    _assert_provider_request(stub, provider, expected_model, "slideshow", max_tokens)


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize(
    "classifier_case",
    [
        ClassifierCase(
            response='{"planet_earth": "yes", "synthetic_media": "no"}',
            classification=["planet_earth"],
            should_filter=False,
        ),
        ClassifierCase(
            response='{"planet_earth": "yes", "synthetic_media": "yes"}',
            classification=["planet_earth", "synthetic_media"],
            should_filter=True,
        ),
    ],
    ids=["allowed", "blocked"],
)
def test_video_classifier_provider_chain_from_split_cli(
    provider: Provider,
    classifier_case: ClassifierCase,
    request: pytest.FixtureRequest,
    video_provider_e2e_context: VideoProviderE2EContext,
) -> None:
    """Classification should retain allowed clips and move blocked clips to filtered output."""
    context = video_provider_e2e_context
    stub = _configure_provider(provider, "classifier", classifier_case.response, request, context.monkeypatch)
    expected_model = "stub-classifier-model" if provider == "openai" else "models/stub-classifier"
    max_tokens = 321
    args = _parse_split_args(
        context.tmp_path,
        [
            "--video-classifier",
            "--video-classifier-endpoint",
            provider,
            "--video-classifier-use-custom-categories",
            "--video-classifier-allow",
            "planet_earth",
            "--video-classifier-block",
            "synthetic_media",
            "--video-classifier-openai-model-name",
            "stub-classifier-model",
            "--video-classifier-openai-retries",
            "1",
            "--video-classifier-openai-retry-delay-seconds",
            "0",
            "--video-classifier-gemini-model-name",
            "models/stub-classifier",
            "--video-classifier-gemini-retries",
            "1",
            "--video-classifier-gemini-retry-delay-seconds",
            "0",
            "--video-classifier-max-output-tokens",
            str(max_tokens),
        ],
    )
    chain = _extract_provider_chain(args, VllmVideoClassifierStage, provider)

    result = context.sequential_runner.run([context.task], chain, "")

    assert result is not None
    assert len(result) == 1
    assert result[0] is context.task
    expected_passed = 0 if classifier_case.should_filter else 1
    expected_filtered = 1 if classifier_case.should_filter else 0
    assert len(context.task.video.clips) == expected_passed
    assert len(context.task.video.filtered_clips) == expected_filtered
    assert context.task.video.clip_stats.num_filtered_by_qwen_classifier == expected_filtered
    clip = context.task.video.filtered_clips[0] if classifier_case.should_filter else context.task.video.clips[0]
    assert clip.qwen_rejection_stage == ("classifier" if classifier_case.should_filter else None)
    assert clip.qwen_type_classification == classifier_case.classification
    assert not clip.errors
    if classifier_case.should_filter:
        assert clip.extracted_frames.resolve() is None
    else:
        assert clip.extracted_frames.resolve() is not None
    assert len(clip.filter_windows) == 1
    window = clip.filter_windows[0]
    assert window.caption[provider] == classifier_case.response
    assert window.caption_status == "success"
    assert window.caption_failure_reason is None
    assert not window.errors
    assert window.mp4_bytes.resolve() is None
    rejection_reasons = window.caption["qwen_rejection_reasons"]
    if classifier_case.should_filter:
        assert rejection_reasons == "{'synthetic_media': 'yes'}"
    else:
        assert rejection_reasons == "{}"
    _assert_provider_request(stub, provider, expected_model, "planet_earth", max_tokens)
