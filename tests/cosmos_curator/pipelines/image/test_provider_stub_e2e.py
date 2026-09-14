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
"""CPU end-to-end coverage for image pipeline provider arms.

The tests parse the annotate CLI, assemble production stages, and run the
resulting load/provider/postprocess chain with ``SequentialRunner``. The final
writer is intentionally omitted so provider outputs can be inspected without
starting Ray or writing duplicate fixture data.
"""

import argparse
import importlib
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Literal, cast

import numpy as np
import pytest

from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini, OpenAIConfig, OpenAIEndpointConfig
from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.pipelines.common import openai_embedding_utils
from cosmos_curator.pipelines.image.annotate_pipeline import _assemble_stages, add_annotate_command
from cosmos_curator.pipelines.image.captioning import image_api_caption_stages
from cosmos_curator.pipelines.image.embedding import image_embedding_stages
from cosmos_curator.pipelines.image.read_write.image_writer_stage import ImageWriterStage
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask
from tests.cosmos_curator.pipelines.provider_stub_utils import GeminiStub, OpenAIStubServer

Provider = Literal["openai", "gemini"]
ProviderStub = OpenAIStubServer | GeminiStub
OpenAIEndpointKey = Literal["caption", "embedding", "filter", "classifier"]

# This coverage is required to run in the CPU unit-test job, which activates the pixi ``dev``
# environment. Other pixi environments still import this module while collecting ``pytest -m env``
# runs, so a missing provider SDK skips there but fails loudly in ``dev`` rather than turning
# required provider coverage into a silent green skip.
UNIT_TEST_PIXI_ENV_NAME = "dev"

# The prep stage re-encodes images as PNG; raw-image fallbacks would use a different media type.
OPENAI_PREP_DATA_URL_PREFIX = "data:image/png;base64,"


@dataclass(frozen=True)
class ImageProviderE2EContext:
    """Shared fixtures for one assembled image provider chain."""

    monkeypatch: pytest.MonkeyPatch
    tmp_path: pathlib.Path
    image_data_dir: pathlib.Path
    task: ImagePipeTask
    sequential_runner: RunnerInterface


@pytest.fixture
def image_provider_e2e_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    image_data_dir: pathlib.Path,
    sample_image_task: ImagePipeTask,
    sequential_runner: RunnerInterface,
) -> ImageProviderE2EContext:
    """Bundle common fixtures so provider tests remain concise."""
    return ImageProviderE2EContext(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        image_data_dir=image_data_dir,
        task=sample_image_task,
        sequential_runner=sequential_runner,
    )


def _require_provider_module(module_name: str) -> ModuleType:
    """Import a provider SDK, failing instead of skipping in the unit-test environment.

    Importing (rather than ``pytest.importorskip``) keeps collection working in the pixi
    environments used by ``pytest -m env`` runs, while guaranteeing that the required
    provider coverage cannot silently disappear from the CPU unit-test job.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        current_env = pixi_utils.get_env_name()
        if current_env == UNIT_TEST_PIXI_ENV_NAME:
            pytest.fail(
                f"{module_name!r} must be installed in the {UNIT_TEST_PIXI_ENV_NAME!r} pixi environment "
                f"so image provider coverage runs in the CPU unit-test job, but importing it failed: {exc}"
            )
        pytest.skip(f"{module_name!r} is not installed in the {current_env or 'active'} environment")
    else:
        return module


def _parse_annotate_args(
    image_data_dir: pathlib.Path,
    output_path: pathlib.Path,
    options: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    return parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(image_data_dir),
            "--output-path",
            str(output_path),
            *options,
        ]
    )


def _run_assembled_chain(
    args: argparse.Namespace,
    task: ImagePipeTask,
    sequential_runner: RunnerInterface,
) -> None:
    """Run every assembled stage except the final image writer."""
    stages = _assemble_stages(args)
    assert all(isinstance(stage, CuratorStageSpec) for stage in stages)
    stage_specs = cast("list[CuratorStageSpec]", stages)
    assert isinstance(stage_specs[-1].stage, ImageWriterStage), (
        f"expected the assembled chain to end with the image writer; got {type(stage_specs[-1].stage).__name__}"
    )

    result = sequential_runner.run([task], stage_specs[:-1], "")

    assert result is not None
    assert len(result) == 1


def _configure_openai(
    monkeypatch: pytest.MonkeyPatch,
    server: OpenAIStubServer,
    endpoint_key: OpenAIEndpointKey,
) -> None:
    """Point production image provider modules at the real SDK and local stub."""
    openai = _require_provider_module("openai")
    embedding_response_module = _require_provider_module("openai.types.create_embedding_response")

    endpoint = OpenAIEndpointConfig(api_key="test-key", base_url=server.base_url)
    endpoint_configs: dict[OpenAIEndpointKey, OpenAIConfig] = {
        "caption": OpenAIConfig(caption=endpoint),
        "embedding": OpenAIConfig(embedding=endpoint),
        "filter": OpenAIConfig(filter=endpoint),
        "classifier": OpenAIConfig(classifier=endpoint),
    }
    config = ConfigFileData(openai=endpoint_configs[endpoint_key])
    monkeypatch.setattr(image_api_caption_stages, "openai", openai, raising=False)
    monkeypatch.setattr(image_api_caption_stages, "maybe_load_config", lambda: config)
    monkeypatch.setattr(image_embedding_stages, "openai", openai, raising=False)
    monkeypatch.setattr(image_embedding_stages, "maybe_load_config", lambda: config)
    monkeypatch.setattr(openai_embedding_utils, "openai", openai, raising=False)
    monkeypatch.setattr(
        openai_embedding_utils,
        "CreateEmbeddingResponse",
        embedding_response_module.CreateEmbeddingResponse,
        raising=False,
    )


def _configure_provider(
    provider: Provider,
    response_text: str,
    endpoint_key: Literal["caption", "filter", "classifier"],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> ProviderStub:
    if provider == "openai":
        server = cast("OpenAIStubServer", request.getfixturevalue("openai_stub_server"))
        _configure_openai(monkeypatch, server, endpoint_key)
        server.set_chat_content(response_text)
        return server

    genai_types = _require_provider_module("google.genai.types")

    factory = cast("Callable[..., GeminiStub]", request.getfixturevalue("gemini_stub"))
    stub = factory(text=response_text)
    monkeypatch.setattr(
        image_api_caption_stages,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="test-key")),
    )
    # ``image_api_caption_stages`` only binds ``genai`` / ``genai_types`` under the pixi
    # "default" env, so both patches must tolerate the attributes being absent elsewhere.
    monkeypatch.setattr(
        image_api_caption_stages,
        "genai",
        SimpleNamespace(Client=lambda **_kwargs: stub.client),
        raising=False,
    )
    monkeypatch.setattr(image_api_caption_stages, "genai_types", genai_types, raising=False)
    return stub


def _assert_provider_request(stub: ProviderStub, provider: Provider, expected_model: str) -> str:
    """Assert the provider request shape and return its outbound text prompt."""
    if provider == "openai":
        assert isinstance(stub, OpenAIStubServer)
        requests = stub.requests_for("/chat/completions")
        assert len(requests) == 1
        body = requests[0].body
        assert body is not None
        assert body["model"] == expected_model
        content = body["messages"][0]["content"]
        assert any(part["type"] == "image_url" for part in content)
        text_parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        ]
        assert text_parts
        return "\n".join(text_parts)

    assert isinstance(stub, GeminiStub)
    assert len(stub.calls) == 1
    assert stub.calls[0]["model"] == expected_model
    contents = stub.calls[0]["contents"]
    parts = contents.get("parts") if isinstance(contents, dict) else getattr(contents, "parts", None)
    assert isinstance(parts, list)
    text_parts = [
        text
        for part in parts
        if isinstance(
            text := (part.get("text") if isinstance(part, dict) else getattr(part, "text", None)),
            str,
        )
    ]
    assert text_parts
    return "\n".join(text_parts)


def _assert_openai_prep_payload(stub: OpenAIStubServer) -> None:
    """Assert the chat request carried the prep stage's PNG payload, not the raw fallback.

    ``ImageOpenAIPrepStage`` swallows failures into ``image.errors`` and
    ``ImageOpenAICaptionStage._resolve_payload`` then silently falls back to the original
    encoded bytes, so only the media type proves the prep stage actually ran.
    """
    requests = stub.requests_for("/chat/completions")
    assert len(requests) == 1
    body = requests[0].body
    assert body is not None
    image_urls = [
        part["image_url"]["url"]
        for part in body["messages"][0]["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert image_urls
    assert all(url.startswith(OPENAI_PREP_DATA_URL_PREFIX) for url in image_urls)


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_image_caption_provider_chain_from_cli_assembly(
    provider: Provider,
    request: pytest.FixtureRequest,
    image_provider_e2e_context: ImageProviderE2EContext,
) -> None:
    """The annotate caption arm should load an image and store provider output."""
    context = image_provider_e2e_context
    prompt_marker = "custom-image-caption-prompt-marker"
    stub = _configure_provider(provider, f"{provider} image caption", "caption", request, context.monkeypatch)
    expected_model = "stub-caption-model" if provider == "openai" else "models/stub-caption"
    args = _parse_annotate_args(
        context.image_data_dir,
        context.tmp_path / "output",
        [
            "--captioning-algorithm",
            provider,
            "--caption-prompt-text",
            prompt_marker,
            "--no-generate-embeddings",
            "--openai-caption-model",
            "stub-caption-model",
            "--openai-caption-retries",
            "1",
            "--openai-retry-delay-seconds",
            "0",
            "--gemini-caption-model",
            "models/stub-caption",
            "--gemini-caption-retries",
            "1",
            "--gemini-retry-delay-seconds",
            "0",
        ],
    )

    _run_assembled_chain(args, context.task, context.sequential_runner)

    assert context.task.image.caption == f"{provider} image caption"
    assert context.task.image.captions[provider] == f"{provider} image caption"
    assert context.task.image.caption_status == "success"
    assert not context.task.image.errors
    prompt = _assert_provider_request(stub, provider, expected_model)
    assert prompt_marker in prompt
    if provider == "openai":
        assert isinstance(stub, OpenAIStubServer)
        _assert_openai_prep_payload(stub)


def test_openai_image_embedding_chain_from_cli_assembly(
    openai_stub_server: OpenAIStubServer,
    image_provider_e2e_context: ImageProviderE2EContext,
) -> None:
    """The annotate embedding arm should store the vector returned by the real SDK."""
    context = image_provider_e2e_context
    _configure_openai(context.monkeypatch, openai_stub_server, "embedding")
    expected_embedding = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    openai_stub_server.set_embedding(expected_embedding.tolist(), model="stub-embed-model")
    args = _parse_annotate_args(
        context.image_data_dir,
        context.tmp_path / "output",
        [
            "--embedding-algorithm",
            "openai",
            "--openai-embedding-model-name",
            "stub-embed-model",
            "--openai-embedding-max-concurrent-requests",
            "1",
            "--no-generate-captions",
        ],
    )

    _run_assembled_chain(args, context.task, context.sequential_runner)

    np.testing.assert_array_equal(context.task.image.embeddings["openai"], expected_embedding)
    requests = openai_stub_server.requests_for("/embeddings")
    assert len(requests) == 1
    body = requests[0].body
    assert body is not None
    assert body["model"] == "stub-embed-model"
    assert body["encoding_format"] == "float"
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_semantic_filter_provider_chain_from_cli_assembly(
    provider: Provider,
    request: pytest.FixtureRequest,
    image_provider_e2e_context: ImageProviderE2EContext,
) -> None:
    """The external semantic-filter arm should run provider inference and postprocess."""
    context = image_provider_e2e_context
    response = '{"synthetic_image": "yes"}'
    stub = _configure_provider(provider, response, "filter", request, context.monkeypatch)
    expected_model = "stub-filter-model" if provider == "openai" else "models/stub-filter"
    args = _parse_annotate_args(
        context.image_data_dir,
        context.tmp_path / "output",
        [
            "--semantic-filter",
            "--semantic-filter-model-variant",
            provider,
            "--semantic-filter-categories",
            "synthetic_image",
            "--semantic-filter-openai-model-name",
            "stub-filter-model",
            "--semantic-filter-openai-retries",
            "1",
            "--semantic-filter-openai-retry-delay-seconds",
            "0",
            "--semantic-filter-gemini-model-name",
            "models/stub-filter",
            "--semantic-filter-gemini-retries",
            "1",
            "--semantic-filter-gemini-retry-delay-seconds",
            "0",
            "--no-generate-embeddings",
            "--no-generate-captions",
        ],
    )

    _run_assembled_chain(args, context.task, context.sequential_runner)

    result_key = f"semantic:{provider}"
    assert context.task.image.filter_captions[result_key] == response
    assert context.task.image.filter_caption_status[result_key] == "success"
    assert context.task.image.is_filtered is True
    assert context.task.image.qwen_rejection_stage == "semantic"
    assert context.task.image.qwen_rejection_reasons == {"synthetic_image": "yes"}
    prompt = _assert_provider_request(stub, provider, expected_model).lower()
    assert "synthetic_image" in prompt
    assert "is there" in prompt


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_image_classifier_provider_chain_from_cli_assembly(
    provider: Provider,
    request: pytest.FixtureRequest,
    image_provider_e2e_context: ImageProviderE2EContext,
) -> None:
    """The external classifier arm should run provider inference and postprocess."""
    context = image_provider_e2e_context
    response = '{"planet_earth": "yes"}'
    stub = _configure_provider(provider, response, "classifier", request, context.monkeypatch)
    expected_model = "stub-classifier-model" if provider == "openai" else "models/stub-classifier"
    args = _parse_annotate_args(
        context.image_data_dir,
        context.tmp_path / "output",
        [
            "--image-classifier",
            "--image-classifier-model-variant",
            provider,
            "--image-classifier-use-custom-categories",
            "--image-classifier-allow",
            "planet_earth",
            "--image-classifier-openai-model-name",
            "stub-classifier-model",
            "--image-classifier-openai-retries",
            "1",
            "--image-classifier-openai-retry-delay-seconds",
            "0",
            "--image-classifier-gemini-model-name",
            "models/stub-classifier",
            "--image-classifier-gemini-retries",
            "1",
            "--image-classifier-gemini-retry-delay-seconds",
            "0",
            "--no-generate-embeddings",
            "--no-generate-captions",
        ],
    )

    _run_assembled_chain(args, context.task, context.sequential_runner)

    result_key = f"classifier:{provider}"
    assert context.task.image.filter_captions[result_key] == response
    assert context.task.image.filter_caption_status[result_key] == "success"
    assert context.task.image.is_filtered is False
    assert context.task.image.qwen_type_classification == ["planet_earth"]
    prompt = _assert_provider_request(stub, provider, expected_model).lower()
    assert "planet_earth" in prompt
    assert "can you answer the following questions about this media" in prompt
