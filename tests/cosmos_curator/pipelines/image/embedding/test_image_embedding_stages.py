# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image embedding stages."""

import pathlib

import numpy as np
import pytest
import torch

from cosmos_curator.core.sensors.data.image_data import ImageData, ImageMetadata
from cosmos_curator.pipelines.image.embedding import image_embedding_stages
from cosmos_curator.pipelines.image.embedding.image_embedding_stages import (
    ImageCLIPEmbeddingStage,
    ImageCosmosEmbed1EmbeddingStage,
    ImageInternVideo2EmbeddingStage,
    ImageOpenAIEmbeddingStage,
)
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask


class _FakeInternVideo2Model:
    """Minimal fake InternVideo2 model for shape-preservation tests."""

    def formulate_input_image(self, frame: np.ndarray) -> np.ndarray:
        return np.expand_dims(frame, axis=0)

    def encode_video_frames(self, _iv2_input: np.ndarray) -> torch.Tensor:
        return torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        return torch.tensor([[float(len(text))]], dtype=torch.float32)

    def evaluate(self, _video_embd: torch.Tensor, _text_embds: list[torch.Tensor]) -> tuple[list[float], list[int]]:
        return [0.95], [0]


class _FakeCosmosEmbed1Model:
    """Minimal fake Cosmos-Embed1 model for text-verification tests."""

    def formulate_input_image(self, frame: np.ndarray) -> np.ndarray:
        return np.expand_dims(frame, axis=0)

    def encode_video_frames(self, _ce1_input: np.ndarray) -> torch.Tensor:
        return torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        return torch.tensor([[float(len(text))]], dtype=torch.float32)

    def evaluate(self, _video_embd: torch.Tensor, _text_embds: list[torch.Tensor]) -> tuple[list[float], list[int]]:
        return [0.96], [0]


class _FakeCLIPModel:
    """Minimal fake CLIP model for image embedding tests."""

    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        assert batch.shape == (1, 8, 8, 3)
        return torch.tensor([[7.0, 8.0]], dtype=torch.float32)


def test_internvideo2_image_embedding_preserves_batch_dimension() -> None:
    """Image InternVideo2 embeddings should match video's single-item batched shape."""
    task = ImagePipeTask(
        session_id="image-1",
        image=Image(
            input_image=pathlib.Path("image-1.jpg"),
            relative_path="image-1.jpg",
            image_data=ImageData.from_frames(
                np.full((1, 8, 8, 3), 127, dtype=np.uint8),
                ImageMetadata(height=8, width=8, image_format="jpg"),
            ),
        ),
    )
    stage = ImageInternVideo2EmbeddingStage()
    stage._model = _FakeInternVideo2Model()  # type: ignore[assignment]

    stage.process_data([task])

    assert task.image.embeddings["internvideo2"].shape == (1, 3)


def test_internvideo2_image_embedding_writes_text_match_when_requested() -> None:
    """Image InternVideo2 stage should expose a best text match when verification texts are provided."""
    task = ImagePipeTask(
        session_id="image-1",
        image=Image(
            input_image=pathlib.Path("image-1.jpg"),
            relative_path="image-1.jpg",
            image_data=ImageData.from_frames(
                np.full((1, 8, 8, 3), 127, dtype=np.uint8),
                ImageMetadata(height=8, width=8, image_format="jpg"),
            ),
        ),
    )
    stage = ImageInternVideo2EmbeddingStage(texts_to_verify=["expected text"])
    stage._model = _FakeInternVideo2Model()  # type: ignore[assignment]

    stage.process_data([task])

    assert task.image.intern_video_2_text_match == ("expected text", 0.95)


def test_cosmos_embed1_image_embedding_writes_text_match_when_requested() -> None:
    """Image Cosmos-Embed1 stage should expose a best text match when verification texts are provided."""
    task = ImagePipeTask(
        session_id="image-1",
        image=Image(
            input_image=pathlib.Path("image-1.jpg"),
            relative_path="image-1.jpg",
            image_data=ImageData.from_frames(
                np.full((1, 8, 8, 3), 127, dtype=np.uint8),
                ImageMetadata(height=8, width=8, image_format="jpg"),
            ),
        ),
    )
    stage = ImageCosmosEmbed1EmbeddingStage(variant="224p", texts_to_verify=["expected text"])
    stage._model = _FakeCosmosEmbed1Model()  # type: ignore[assignment]

    stage.process_data([task])

    assert task.image.cosmos_embed1_text_match == ("expected text", 0.96)


def test_clip_image_embedding_writes_embedding() -> None:
    """Image CLIP stage should write the embedding returned by the model."""
    task = ImagePipeTask(
        session_id="image-1",
        image=Image(
            input_image=pathlib.Path("image-1.jpg"),
            relative_path="image-1.jpg",
            image_data=ImageData.from_frames(
                np.full((1, 8, 8, 3), 127, dtype=np.uint8),
                ImageMetadata(height=8, width=8, image_format="jpg"),
            ),
        ),
    )
    stage = ImageCLIPEmbeddingStage()
    stage._model = _FakeCLIPModel()  # type: ignore[assignment]

    stage.process_data([task])

    np.testing.assert_array_equal(task.image.embeddings["clip"], np.array([7.0, 8.0], dtype=np.float32))


def test_openai_image_embedding_writes_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image OpenAI stage should store the embedding returned by the API helper."""
    task = ImagePipeTask(
        session_id="image-1",
        image=Image(
            input_image=pathlib.Path("image-1.jpg"),
            relative_path="image-1.jpg",
            image_data=ImageData.from_frames(
                np.full((1, 8, 8, 3), 127, dtype=np.uint8),
                ImageMetadata(height=8, width=8, image_format="jpg"),
            ),
        ),
    )
    stage = ImageOpenAIEmbeddingStage(model_name="fake-model", max_concurrent_requests=1)
    stage._client = object()  # type: ignore[assignment]

    def _fake_call_openai_embedding_api(
        _client: object,
        model_name: str,
        content_parts: list[dict[str, object]],
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> np.ndarray:
        assert model_name == "fake-model"
        assert max_retries == 3
        assert retry_delay_seconds == 1.0
        assert content_parts[0]["type"] == "image_url"
        image_url = content_parts[0]["image_url"]
        assert isinstance(image_url, dict)
        assert str(image_url["url"]).startswith("data:image/jpeg;base64,")
        return np.array([9.0, 10.0, 11.0], dtype=np.float32)

    monkeypatch.setattr(image_embedding_stages, "call_openai_embedding_api", _fake_call_openai_embedding_api)

    stage.process_data([task])

    np.testing.assert_array_equal(task.image.embeddings["openai"], np.array([9.0, 10.0, 11.0], dtype=np.float32))
