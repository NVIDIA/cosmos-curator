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

"""Enhance caption tests for ChatLM variants."""

import pathlib
import uuid

import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.utils.config.config import ConfigFileData, OpenAIConfig, OpenAIEndpointConfig
from cosmos_curator.pipelines.video.captioning.captioning_stages import (  # type: ignore[import-untyped]
    EnhanceCaptionStage,
)
from cosmos_curator.pipelines.video.utils.data_model import (  # type: ignore[import-untyped]
    Clip,
    SplitPipeTask,
    Video,
    Window,
)


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", ["qwen_lm", "gpt_oss_20b", "openai"])
def test_enhance_caption_lm_variants(
    model_variant: str,
    sequential_runner: RunnerInterface,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnhanceCaptionStage with real LM and pre-filled captions (no prior stages)."""
    base_captions = [
        "A red pickup truck is parked on a cobblestone street.",
        "Interior car shot with a driver speaking into a microphone.",
    ]
    clips: list[Clip] = []
    base_caption_key = "caption_model_variant"
    for i, text in enumerate(base_captions):
        clip = Clip(
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"sample_video.mp4#fake-{i}"),
            source_video="sample_video.mp4",
            span=(0.0, 5.0),
        )
        clip.windows.append(
            Window(
                start_frame=0,
                end_frame=256,
            )
        )
        clip.windows[0].caption[base_caption_key] = text
        clips.append(clip)

    video = Video(input_video=pathlib.Path("sample_video.mp4"), clips=clips)
    task = SplitPipeTask(session_id="test-session", video=video)

    if model_variant == "openai":
        monkeypatch.setattr(
            "cosmos_curator.models.chat_lm.maybe_load_config",
            lambda: ConfigFileData(
                openai=OpenAIConfig(enhance=OpenAIEndpointConfig(api_key="fake-key", base_url="https://fake.endpoint"))
            ),
        )

        def _fake_generate_remote(
            self: object,
            prompts: list[list[dict[str, str]]],
            batch_size: int | None,
        ) -> list[str]:
            del self
            del batch_size
            return [
                f"This is a remote enhanced caption with additional description {idx}"
                for idx, _prompt in enumerate(prompts, start=1)
            ]

        monkeypatch.setattr(
            "cosmos_curator.models.chat_lm.ChatLM._generate_remote",
            _fake_generate_remote,
        )
        monkeypatch.setattr(
            "cosmos_curator.models.chat_lm.resolve_model_name_auto",
            lambda _client, model, **_kwargs: model,
        )

    stages = [EnhanceCaptionStage(model_variant=model_variant)]
    result_tasks = run_pipeline([task], stages, runner=sequential_runner)

    assert result_tasks is not None
    assert len(result_tasks) == 1

    # Verify enhanced captions were written back and are longer than base
    for i, c in enumerate(result_tasks[0].video.clips):
        assert model_variant in c.windows[0].enhanced_caption
        enhanced = c.windows[0].enhanced_caption[model_variant]
        assert isinstance(enhanced, str)
        assert len(enhanced) > len(base_captions[i])


def test_openai_variant_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote enhance caption variant should fail fast without credentials."""
    monkeypatch.setattr("cosmos_curator.models.chat_lm.maybe_load_config", lambda: ConfigFileData(openai=None))

    with pytest.raises(
        RuntimeError,
        match="OpenAI enhance configuration not found",
    ):
        EnhanceCaptionStage(model_variant="openai")
