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

"""Pinned Cosmos preprocessing and preservation of model-normalized projections."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.multimodal_embedding import model_adapter
from cosmos_curator.next.recipes.multimodal_embedding.contracts import resolve_model
from cosmos_curator.next.recipes.multimodal_embedding.model_adapter import validate_projection
from cosmos_curator.next.recipes.multimodal_embedding.video_media import VideoMediaError


def test_projection_cast_preserves_bf16_normalized_values() -> None:
    """BF16 rounding error is tolerated without renormalizing the vector."""
    model_projection = np.zeros((1, 768), dtype=np.float64)
    model_projection[0, 0] = 0.99609375
    result = validate_projection(model_projection, dimension=768)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, model_projection[0].astype(np.float32))


@pytest.mark.parametrize(
    "invalid_projection", [np.zeros((1, 768)), np.full((1, 768), np.nan), np.ones((2, 768)), np.ones((1, 768))]
)
def test_invalid_projection_is_an_item_error(invalid_projection: np.ndarray) -> None:
    """Invalid shapes, values, and norms are rejected without repairing the output."""
    with pytest.raises(VideoMediaError) as exc_info:
        validate_projection(invalid_projection, dimension=768)
    assert exc_info.value.code == "invalid_vector"


@pytest.mark.parametrize("variant", ["cosmos_embed1_224p", "cosmos_embed1_336p", "cosmos_embed1_448p"])
def test_cosmos_processor_receives_unscaled_uint8_rgb_btchw(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The selected pinned processor owns resize/scaling and receives unmodified uint8 RGB."""
    model = resolve_model(variant)
    pixels = np.zeros((8, 3, model.resolution, model.resolution), dtype=np.float32)
    tensor = Mock()
    tensor.__getitem__ = Mock(return_value=SimpleNamespace(float=lambda: SimpleNamespace(numpy=lambda: pixels)))
    processor = Mock(num_video_frames=8, resolution=model.resolution, return_value={"videos": tensor})
    loader = Mock(return_value=processor)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=loader))
    )
    set_threads = Mock()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=set_threads))
    frames = tuple(np.full((7, 11, 3), index * 25, dtype=np.uint8) for index in range(8))
    prepared = model_adapter.load_video_preprocessor(model, cpus=2).preprocess(frames)
    set_threads.assert_called_once_with(2)
    inputs = processor.call_args.kwargs
    assert set(inputs) == {"videos"}
    np.testing.assert_array_equal(inputs["videos"], np.stack(frames)[None].transpose(0, 1, 4, 2, 3))
    assert inputs["videos"].dtype == np.uint8
    assert prepared.shape == (8, 3, model.resolution, model.resolution)
    loader.assert_called_once_with(str(model.runtime_model_dir), trust_remote_code=True, local_files_only=True)


@pytest.mark.parametrize("variant", ["cosmos_embed1_224p", "cosmos_embed1_336p", "cosmos_embed1_448p"])
def test_cosmos_encoder_dispatch(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """All model bindings use the same Cosmos adapter."""
    encoder = object()
    monkeypatch.setattr(model_adapter, "CosmosEmbed1VideoEncoder", lambda _model: encoder)
    assert model_adapter.load_video_encoder(resolve_model(variant)) is encoder


@pytest.fixture
def mock_cosmos_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace optional GPU dependencies while exercising the real adapter constructors."""
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", tmp_path)
    spec = resolve_model("cosmos_embed1_336p")
    spec.runtime_model_dir.mkdir(parents=True)
    torch = SimpleNamespace(
        set_num_threads=Mock(),
        cuda=SimpleNamespace(is_available=Mock(return_value=True)),
        bfloat16=object(),
        from_numpy=Mock(),
    )
    processor = Mock(num_video_frames=8, resolution=336)
    processor_loader = Mock(return_value=processor)
    model = Mock(config=SimpleNamespace(embed_dim=768))
    model.to.return_value = model
    model_loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    return SimpleNamespace(spec=spec, torch=torch, processor=processor, model=model, model_loader=model_loader)


@pytest.mark.parametrize(("attribute", "value"), [("num_video_frames", 4), ("resolution", 224)])
def test_processor_contract_mismatch_is_rejected(
    mock_cosmos_runtime: SimpleNamespace, attribute: str, value: int
) -> None:
    """A differently staged processor cannot silently change the pinned preprocessing."""
    setattr(mock_cosmos_runtime.processor, attribute, value)
    with pytest.raises(ValueError, match="registered contract"):
        model_adapter.load_video_preprocessor(mock_cosmos_runtime.spec)


def test_processor_rejects_wrong_selected_frame_count(mock_cosmos_runtime: SimpleNamespace) -> None:
    """Wrong frame counts fail before the processor is called."""
    processor = model_adapter.load_video_preprocessor(mock_cosmos_runtime.spec)
    with pytest.raises(ValueError, match="Expected 8 selected RGB frames, got 1"):
        processor.preprocess((np.zeros((2, 2, 3), dtype=np.uint8),))
    mock_cosmos_runtime.processor.assert_not_called()


def test_processor_rejects_wrong_output_shape(mock_cosmos_runtime: SimpleNamespace) -> None:
    """The stage boundary rejects tensors that differ from the selected model shape."""
    tensor = Mock()
    tensor.__getitem__ = Mock(
        return_value=SimpleNamespace(float=lambda: SimpleNamespace(numpy=lambda: np.zeros((8, 3, 224, 224))))
    )
    mock_cosmos_runtime.processor.return_value = {"videos": tensor}
    processor = model_adapter.load_video_preprocessor(mock_cosmos_runtime.spec)
    with pytest.raises(ValueError, match="unexpected video tensor"):
        processor.preprocess(tuple(np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(8)))


def test_encoder_requires_cuda_before_loading_weights(mock_cosmos_runtime: SimpleNamespace) -> None:
    """A node without a visible GPU fails without attempting checkpoint loading."""
    mock_cosmos_runtime.torch.cuda.is_available.return_value = False
    with pytest.raises(RuntimeError, match="visible CUDA GPU"):
        model_adapter.load_video_encoder(mock_cosmos_runtime.spec)
    mock_cosmos_runtime.model_loader.assert_not_called()


def test_encoder_requires_model_directory_before_loading_weights(mock_cosmos_runtime: SimpleNamespace) -> None:
    """Actor-local staging is checked even when driver-side preflight previously passed."""
    mock_cosmos_runtime.spec.runtime_model_dir.rmdir()
    with pytest.raises(FileNotFoundError, match="model weights are missing"):
        model_adapter.load_video_encoder(mock_cosmos_runtime.spec)
    mock_cosmos_runtime.model_loader.assert_not_called()


def test_encoder_rejects_model_dimension_mismatch(mock_cosmos_runtime: SimpleNamespace) -> None:
    """Loading another projection dimension cannot produce a compatible vector field."""
    mock_cosmos_runtime.model.config.embed_dim = 256
    with pytest.raises(ValueError, match="model dimension differs"):
        model_adapter.load_video_encoder(mock_cosmos_runtime.spec)
    mock_cosmos_runtime.model.get_video_embeddings.assert_not_called()


@pytest.mark.parametrize("shape", [(8, 3, 2, 2), (1, 4, 3, 2, 2)])
def test_encoder_rejects_batch_shape_before_gpu_transfer(
    mock_cosmos_runtime: SimpleNamespace, shape: tuple[int, ...]
) -> None:
    """Batch dimensionality and frame count are checked before GPU work begins."""
    encoder = model_adapter.load_video_encoder(mock_cosmos_runtime.spec)
    with pytest.raises(ValueError, match="incompatible shape"):
        encoder.encode_batch(np.zeros(shape, dtype=np.float32))
    mock_cosmos_runtime.torch.from_numpy.assert_not_called()
    mock_cosmos_runtime.model.get_video_embeddings.assert_not_called()
