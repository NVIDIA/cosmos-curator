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

"""Test the Cosmos-Embed1 embedding result."""

import pathlib
import uuid

import pytest
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import ClipTranscodingStage
from cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages import ClipFrameExtractionStage
from cosmos_curator.pipelines.video.embedding.cosmos_embed1_stages import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import FrameExtractionPolicy

_MATCH_CONFIDENCE_SCORE = 0.9


def _get_texts() -> list[str]:
    return [
        "An animated character in a fantasy scene with dramatic lighting",
        "A person swimming in a pool at a sports center",
        "A car driving on a highway through the desert",
        "A dog playing fetch in a park on a sunny day",
    ]


@pytest.fixture
def sample_embedding_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample embedding task."""
    clip = Clip(
        uuid=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        source_video="sample_video.mp4",
        span=(2.5, 6.5),
    )
    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        encoded_data=sample_video_data,
        clips=[clip],
    )
    return SplitPipeTask(
        session_id="test-session",
        video=video,
    )


@pytest.mark.env("legacy-transformers")
def test_generate_embedding(sample_embedding_task: SplitPipeTask, sequential_runner: RunnerInterface) -> None:
    """Test the Cosmos-Embed1Embedding result."""
    stages = [
        ClipTranscodingStage(encoder="libopenh264"),
        ClipFrameExtractionStage(extraction_policies=(FrameExtractionPolicy.sequence,), target_fps=[2.0]),
        CosmosEmbed1FrameCreationStage(target_fps=2.0),
        CosmosEmbed1EmbeddingStage(num_gpus_per_worker=1.0, texts_to_verify=_get_texts()),
    ]
    tasks = run_pipeline([sample_embedding_task], stages, runner=sequential_runner)
    result_task = tasks[0]
    clip = result_task.video.clips[0]

    assert clip.encoded_data.resolve() is not None, "Expected clip.encoded_data to be not None, but it is None."
    assert clip.cosmos_embed1_embedding is not None, "Expected Cosmos-Embed1 embedding to be not None, but it is None."

    text_match = clip.cosmos_embed1_text_match

    assert text_match is not None, "Expected Cosmos-Embed1 text match to be not None, but it is None."

    logger.info(f"Best text match (score={text_match[1]}): {text_match[0]}")

    assert text_match[0] in _get_texts(), f"Text match [{text_match[0]}] not in provided texts"
    assert text_match[1] >= _MATCH_CONFIDENCE_SCORE, (
        f"Expected text match score >= {_MATCH_CONFIDENCE_SCORE}, got {text_match[1]}"
    )

    logger.info("Cosmos-Embed1 embedding test passed.")
