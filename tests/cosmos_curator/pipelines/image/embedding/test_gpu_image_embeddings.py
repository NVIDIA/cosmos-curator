# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for image embedding stages."""

import pathlib

import pytest
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.image.embedding.image_embedding_stages import (
    ImageCosmosEmbed1EmbeddingStage,
    ImageInternVideo2EmbeddingStage,
)
from cosmos_curator.pipelines.image.read_write.image_load_stage import ImageLoadStage
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask

_MATCH_CONFIDENCE_SCORE = 0.9


def _get_texts() -> list[str]:
    return [
        "An animated character in a fantasy scene with dramatic lighting",
        "A person swimming in a pool at a sports center",
        "A car driving on a highway through the desert",
        "A dog playing fetch in a park on a sunny day",
        "An animated action scene with swords and combat",
    ]


@pytest.mark.env("legacy-transformers")
def test_image_internvideo2_embedding_generation(
    sample_image_task: ImagePipeTask,
    sequential_runner: RunnerInterface,
    image_data_dir: pathlib.Path,
) -> None:
    """Run image pipeline (load -> InternVideo2 embed) and assert embedding + text-match sanity."""
    stages = [
        ImageLoadStage(
            input_path=str(image_data_dir),
            input_s3_profile_name="default",
            verbose=False,
            log_stats=False,
        ),
        ImageInternVideo2EmbeddingStage(
            num_gpus_per_worker=1.0,
            texts_to_verify=_get_texts(),
        ),
    ]
    tasks = run_pipeline([sample_image_task], stages, runner=sequential_runner)

    assert tasks is not None
    assert len(tasks) == 1
    image = tasks[0].image
    assert "internvideo2" in image.embeddings
    assert image.embeddings["internvideo2"].size > 0
    assert image.intern_video_2_text_match is not None

    text_match = image.intern_video_2_text_match
    assert text_match[0] in _get_texts()
    logger.info(f"Best image InternVideo2 text match (score={text_match[1]}): {text_match[0]}")
    assert text_match[1] >= _MATCH_CONFIDENCE_SCORE


@pytest.mark.env("legacy-transformers")
def test_image_cosmos_embed1_embedding_generation(
    sample_image_task: ImagePipeTask,
    sequential_runner: RunnerInterface,
    image_data_dir: pathlib.Path,
) -> None:
    """Run image pipeline (load -> Cosmos-Embed1 embed) and assert embedding + text-match sanity."""
    stages = [
        ImageLoadStage(
            input_path=str(image_data_dir),
            input_s3_profile_name="default",
            verbose=False,
            log_stats=False,
        ),
        ImageCosmosEmbed1EmbeddingStage(
            variant="224p",
            num_gpus_per_worker=1.0,
            texts_to_verify=_get_texts(),
        ),
    ]
    tasks = run_pipeline([sample_image_task], stages, runner=sequential_runner)

    assert tasks is not None
    assert len(tasks) == 1
    image = tasks[0].image
    assert "cosmos_embed1_224p" in image.embeddings
    assert image.embeddings["cosmos_embed1_224p"].size > 0
    assert image.cosmos_embed1_text_match is not None

    text_match = image.cosmos_embed1_text_match
    assert text_match[0] in _get_texts()
    logger.info(f"Best image Cosmos-Embed1 text match (score={text_match[1]}): {text_match[0]}")
    assert text_match[1] >= _MATCH_CONFIDENCE_SCORE
