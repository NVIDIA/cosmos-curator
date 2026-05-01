# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image filter post-processing stages."""

import pathlib

import numpy as np

from cosmos_curator.pipelines.image.filtering.filter_stages import ImageClassifierStage, ImageSemanticFilterStage
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask


def _make_task() -> ImagePipeTask:
    path = pathlib.Path("/fake/image.jpg")
    image = Image(
        input_image=path,
        relative_path="image.jpg",
        encoded_data=np.frombuffer(b"\xff\xd8\xff\xdbraw", dtype=np.uint8),
    )
    return ImagePipeTask(session_id=str(path), image=image)


def test_semantic_filter_stage_preserves_external_blocked_status() -> None:
    """Blocked endpoint filter results should not be rewritten as prep failures."""
    task = _make_task()
    task.image.filter_caption_status["semantic:openai"] = "blocked"

    ImageSemanticFilterStage(model_variant="openai", filter_caption_key="semantic:openai").process_data([task])

    assert task.image.is_filtered is True
    assert task.image.qwen_rejection_stage == "semantic"
    assert "openai" not in task.image.errors


def test_semantic_filter_stage_score_only_preserves_external_error_without_filtering() -> None:
    """Score-only semantic filtering should keep external errors without rejecting the image."""
    task = _make_task()
    task.image.filter_caption_status["semantic:gemini"] = "error"
    task.image.filter_caption_failure_reason["semantic:gemini"] = "exception"
    task.image.errors["semantic:gemini_filter_caption"] = "upstream API failure"

    ImageSemanticFilterStage(
        model_variant="gemini",
        filter_caption_key="semantic:gemini",
        score_only=True,
    ).process_data([task])

    assert task.image.is_filtered is False
    assert task.image.qwen_rejection_stage is None
    assert task.image.errors["semantic:gemini_filter_caption"] == "upstream API failure"
    assert "gemini" not in task.image.errors


def test_classifier_stage_preserves_external_error_status() -> None:
    """Classifier postprocess should preserve endpoint errors instead of reporting prep failure."""
    task = _make_task()
    task.image.filter_caption_status["classifier:openai"] = "error"
    task.image.filter_caption_failure_reason["classifier:openai"] = "exception"
    task.image.errors["classifier:openai_filter_caption"] = "bad key"

    ImageClassifierStage(
        model_variant="openai",
        filter_caption_key="classifier:openai",
        custom_categories=True,
        type_allow="planet_earth",
    ).process_data([task])

    assert task.image.is_filtered is True
    assert task.image.qwen_rejection_stage == "classifier"
    assert task.image.errors["classifier:openai_filter_caption"] == "bad key"
    assert "openai" not in task.image.errors
