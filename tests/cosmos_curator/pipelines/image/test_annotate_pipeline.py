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

"""Tests for the builder-based image annotate pipeline."""

import argparse
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.pipelines.image.annotate_pipeline import _assemble_stages, add_annotate_command, write_summary
from cosmos_curator.pipelines.image.captioning import image_api_caption_stages
from cosmos_curator.pipelines.image.captioning.image_api_caption_stages import (
    ImageGeminiCaptionStage,
    ImageOpenAICaptionStage,
    ImageOpenAIPrepStage,
)
from cosmos_curator.pipelines.image.captioning.image_vllm_stages import ImageVllmCaptionStage, ImageVllmPrepStage
from cosmos_curator.pipelines.image.embedding.image_embedding_stages import ImageInternVideo2EmbeddingStage
from cosmos_curator.pipelines.image.filtering.filter_stages import ImageClassifierStage, ImageSemanticFilterStage
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask


def test_add_annotate_command_registers_subcommand(tmp_path: pathlib.Path) -> None:
    """The annotate CLI should parse the basic required image pipeline arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)

    args = parser.parse_args(["annotate", "--input-image-path", str(tmp_path / "in"), "--output-path", str(tmp_path)])
    assert args.command == "annotate"
    assert callable(args.func)


def test_assemble_stages_without_captioning_returns_ingest_embedding_and_output(tmp_path: pathlib.Path) -> None:
    """Disabling captions should still leave ingest, embedding, and output stage specs."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--no-generate-captions",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 3
    assert all(isinstance(stage, CuratorStageSpec) for stage in stages)
    assert isinstance(stages[1].stage, ImageInternVideo2EmbeddingStage)


def test_assemble_stages_with_captioning_returns_five_specs(tmp_path: pathlib.Path) -> None:
    """Enabling captions should return ingest, embedding, prep, caption, and output stage specs."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--captioning-algorithm",
            "qwen",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 5
    assert all(isinstance(stage, CuratorStageSpec) for stage in stages)
    assert isinstance(stages[1].stage, ImageInternVideo2EmbeddingStage)


def test_assemble_stages_with_gemini_captioning_returns_embedding_and_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini should build ingest, embedding, Gemini caption, and output stages."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--captioning-algorithm",
            "gemini",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 4
    assert isinstance(stages[1].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[2].stage, ImageGeminiCaptionStage)
    assert stages[2].stage.stage_batch_size == args.caption_batch_size


def test_assemble_stages_with_openai_captioning_returns_five_specs(tmp_path: pathlib.Path) -> None:
    """OpenAI should default to ingest, embedding, prep, caption, and output stages."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--captioning-algorithm",
            "openai",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 5
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[2].stage, ImageOpenAIPrepStage)
    assert isinstance(stages[3].stage, ImageOpenAICaptionStage)
    assert stages[3].stage.stage_batch_size == args.caption_batch_size


def test_assemble_stages_with_openai_raw_image_skips_prep(tmp_path: pathlib.Path) -> None:
    """OpenAI raw-image mode should keep embedding before caption and skip prep."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--captioning-algorithm",
            "openai",
            "--openai-caption-raw-image",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 4
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[2].stage, ImageOpenAICaptionStage)


def test_assemble_stages_with_local_semantic_filter_returns_filter_specs(tmp_path: pathlib.Path) -> None:
    """Local semantic filtering should run before the main embedding and caption stages."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--semantic-filter",
            "enable",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 8
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[3], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageVllmPrepStage)
    assert isinstance(stages[2].stage, ImageVllmCaptionStage)
    assert isinstance(stages[3].stage, ImageSemanticFilterStage)
    assert isinstance(stages[4], CuratorStageSpec)
    assert isinstance(stages[5], CuratorStageSpec)
    assert isinstance(stages[6], CuratorStageSpec)
    assert isinstance(stages[4].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[5].stage, ImageVllmPrepStage)
    assert isinstance(stages[6].stage, ImageVllmCaptionStage)
    assert stages[2].stage._result_target == "filter_caption"
    assert stages[6].stage._result_target == "caption"


def test_assemble_stages_with_openai_semantic_filter_returns_endpoint_specs(tmp_path: pathlib.Path) -> None:
    """OpenAI semantic filtering should run endpoint stages before normal captioning."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--semantic-filter",
            "enable",
            "--semantic-filter-model-variant",
            "openai",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 8
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[3], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageOpenAIPrepStage)
    assert isinstance(stages[2].stage, ImageOpenAICaptionStage)
    assert stages[2].stage._endpoint_key == "filter"
    assert stages[2].stage.stage_batch_size == args.semantic_filter_batch_size
    assert stages[2].stage._result_target == "filter_caption"
    assert isinstance(stages[3].stage, ImageSemanticFilterStage)
    assert isinstance(stages[4].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[5].stage, ImageVllmPrepStage)
    assert isinstance(stages[6].stage, ImageVllmCaptionStage)


def test_assemble_stages_with_gemini_semantic_filter_returns_endpoint_specs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini semantic filtering should run endpoint stages before normal captioning."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--semantic-filter",
            "enable",
            "--semantic-filter-model-variant",
            "gemini",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 7
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageGeminiCaptionStage)
    assert stages[1].stage.stage_batch_size == args.semantic_filter_batch_size
    assert stages[1].stage._result_target == "filter_caption"
    assert isinstance(stages[2].stage, ImageSemanticFilterStage)
    assert isinstance(stages[3].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[4].stage, ImageVllmPrepStage)
    assert isinstance(stages[5].stage, ImageVllmCaptionStage)


def test_assemble_stages_with_local_classifier_returns_classifier_specs(tmp_path: pathlib.Path) -> None:
    """Local image classifier should run before the main embedding and caption stages."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--image-classifier",
            "enable",
            "--image-classifier-use-custom-categories",
            "--image-classifier-allow",
            "planet_earth",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 8
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[3], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageVllmPrepStage)
    assert isinstance(stages[2].stage, ImageVllmCaptionStage)
    assert isinstance(stages[3].stage, ImageClassifierStage)
    assert isinstance(stages[4], CuratorStageSpec)
    assert isinstance(stages[5], CuratorStageSpec)
    assert isinstance(stages[6], CuratorStageSpec)
    assert isinstance(stages[4].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[5].stage, ImageVllmPrepStage)
    assert isinstance(stages[6].stage, ImageVllmCaptionStage)
    assert stages[2].stage._result_target == "filter_caption"
    assert stages[6].stage._result_target == "caption"


def test_assemble_stages_with_openai_classifier_returns_endpoint_specs(tmp_path: pathlib.Path) -> None:
    """OpenAI classifier should run endpoint stages before normal captioning."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--image-classifier",
            "enable",
            "--image-classifier-model-variant",
            "openai",
            "--image-classifier-use-custom-categories",
            "--image-classifier-allow",
            "planet_earth",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 8
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[3], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageOpenAIPrepStage)
    assert isinstance(stages[2].stage, ImageOpenAICaptionStage)
    assert stages[2].stage._endpoint_key == "classifier"
    assert stages[2].stage.stage_batch_size == args.image_classifier_batch_size
    assert stages[2].stage._result_target == "filter_caption"
    assert isinstance(stages[3].stage, ImageClassifierStage)
    assert isinstance(stages[4].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[5].stage, ImageVllmPrepStage)
    assert isinstance(stages[6].stage, ImageVllmCaptionStage)


def test_assemble_stages_with_gemini_classifier_returns_endpoint_specs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini classifier should run endpoint stages before normal captioning."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--image-classifier",
            "enable",
            "--image-classifier-model-variant",
            "gemini",
            "--image-classifier-use-custom-categories",
            "--image-classifier-allow",
            "planet_earth",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 7
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageGeminiCaptionStage)
    assert stages[1].stage.stage_batch_size == args.image_classifier_batch_size
    assert stages[1].stage._result_target == "filter_caption"
    assert isinstance(stages[2].stage, ImageClassifierStage)
    assert isinstance(stages[3].stage, ImageInternVideo2EmbeddingStage)
    assert isinstance(stages[4].stage, ImageVllmPrepStage)
    assert isinstance(stages[5].stage, ImageVllmCaptionStage)


def test_assemble_stages_builds_openai_api_stages(tmp_path: pathlib.Path) -> None:
    """OpenAI image API stages should still be assembled across caption/filter/classifier flows."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--captioning-algorithm",
            "openai",
            "--semantic-filter",
            "enable",
            "--semantic-filter-model-variant",
            "openai",
            "--image-classifier",
            "enable",
            "--image-classifier-model-variant",
            "openai",
            "--image-classifier-use-custom-categories",
            "--image-classifier-allow",
            "planet_earth",
        ]
    )

    stages = _assemble_stages(args)
    openai_stages = [
        stage.stage
        for stage in stages
        if isinstance(stage, CuratorStageSpec) and isinstance(stage.stage, ImageOpenAICaptionStage)
    ]

    assert openai_stages


def test_assemble_stages_builds_gemini_api_stages(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini image API stages should still be assembled across caption/filter/classifier flows."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--captioning-algorithm",
            "gemini",
            "--semantic-filter",
            "enable",
            "--semantic-filter-model-variant",
            "gemini",
            "--image-classifier",
            "enable",
            "--image-classifier-model-variant",
            "gemini",
            "--image-classifier-use-custom-categories",
            "--image-classifier-allow",
            "planet_earth",
        ]
    )

    stages = _assemble_stages(args)
    gemini_stages = [
        stage.stage
        for stage in stages
        if isinstance(stage, CuratorStageSpec) and isinstance(stage.stage, ImageGeminiCaptionStage)
    ]

    assert gemini_stages


def test_write_summary_includes_embedding_fields(tmp_path: pathlib.Path) -> None:
    """summary.json should report embedding backend and number of images with embeddings."""
    output_path = str(tmp_path)
    args = SimpleNamespace(
        output_path=output_path,
        output_s3_profile_name="",
        perf_profile=False,
        caption_prep_min_pixels=None,
        caption_prep_max_pixels=None,
        embedding_algorithm="internvideo2",
        generate_embeddings=True,
    )
    tasks = [
        ImagePipeTask(
            session_id="image-1",
            image=Image(
                input_image=pathlib.Path("image-1.jpg"),
                relative_path="image-1.jpg",
                encoded_data=LazyData.coerce(np.frombuffer(b"abc", dtype=np.uint8)),
                embeddings={"internvideo2": np.array([[1.0, 2.0]], dtype=np.float32)},
            ),
        ),
        ImagePipeTask(
            session_id="image-2",
            image=Image(
                input_image=pathlib.Path("image-2.jpg"),
                relative_path="image-2.jpg",
                encoded_data=LazyData.coerce(np.frombuffer(b"def", dtype=np.uint8)),
            ),
        ),
    ]

    write_summary(args, num_tasks=2, output_tasks=tasks, pipeline_run_time_min=1.25)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["embedding_backend"] == "internvideo2"
    assert summary["num_images_with_embeddings"] == 1


def test_write_summary_omits_embedding_backend_when_disabled(tmp_path: pathlib.Path) -> None:
    """summary.json should set embedding_backend to null when embedding generation is disabled."""
    output_path = str(tmp_path)
    args = SimpleNamespace(
        output_path=output_path,
        output_s3_profile_name="",
        perf_profile=False,
        caption_prep_min_pixels=None,
        caption_prep_max_pixels=None,
        embedding_algorithm="internvideo2",
        generate_embeddings=False,
    )
    tasks = [
        ImagePipeTask(
            session_id="image-1",
            image=Image(
                input_image=pathlib.Path("image-1.jpg"),
                relative_path="image-1.jpg",
                encoded_data=LazyData.coerce(np.frombuffer(b"abc", dtype=np.uint8)),
            ),
        )
    ]

    write_summary(args, num_tasks=1, output_tasks=tasks, pipeline_run_time_min=0.5)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["embedding_backend"] is None
    assert summary["num_images_with_embeddings"] == 0
