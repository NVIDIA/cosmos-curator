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

"""CPU frame preprocessing and GPU model inference for pinned video variants."""

from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from cosmos_curator.next.recipes.multimodal_embedding.contracts import VECTOR_NORM_TOLERANCE, VideoEmbeddingModelSpec
from cosmos_curator.next.recipes.multimodal_embedding.video_media import VideoMediaError

_INVALID_VECTOR = "invalid_vector"
_VIDEO_BATCH_NDIM = 5


class VideoEncoder(Protocol):
    """One loaded GPU adapter from a batch of prepared videos to projections."""

    def encode_batch(self, videos: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return one model-normalized projection per prepared video."""
        ...


class VideoPreprocessor(Protocol):
    """One CPU adapter from selected RGB frames to a compact model tensor."""

    def preprocess(self, frames: tuple[npt.NDArray[np.uint8], ...]) -> npt.NDArray[np.float32]:
        """Prepare one video with the model's pinned processor."""
        ...


def validate_projection(values: object, *, dimension: int) -> npt.NDArray[np.float32]:
    """Cast the model-normalized projection to float32 without changing its values."""
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        msg = "Model returned a projection that cannot be converted to float32"
        raise VideoMediaError(_INVALID_VECTOR, msg) from exc
    if array.shape != (1, dimension) or not np.isfinite(array).all():
        msg = f"Model must return one finite projection of dimension {dimension}"
        raise VideoMediaError(_INVALID_VECTOR, msg)
    if not np.isclose(np.linalg.norm(array[0].astype(np.float64)), 1.0, atol=VECTOR_NORM_TOLERANCE):
        msg = "Model vector is outside the unit-norm tolerance"
        raise VideoMediaError(_INVALID_VECTOR, msg)
    return cast("npt.NDArray[np.float32]", array[0])


class CosmosEmbed1VideoPreprocessor:
    """Run the checkpoint's processor on CPU before crossing the Ray stage boundary."""

    def __init__(self, model: VideoEmbeddingModelSpec, *, cpus: int = 1) -> None:
        """Load the registered checkpoint processor without GPU model weights."""
        import torch  # noqa: PLC0415 -- legacy-transformers CPU actor only
        from transformers import AutoProcessor  # noqa: PLC0415 -- legacy-transformers media actor only

        torch.set_num_threads(cpus)
        self._processor: Any = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(model.runtime_model_dir), trust_remote_code=True, local_files_only=True
        )
        if (
            self._processor.num_video_frames != model.video_frame_count
            or self._processor.resolution != model.resolution
        ):
            msg = "Cosmos-Embed1 processor frame count differs from the registered contract"
            raise ValueError(msg)
        self._spec = model

    def preprocess(self, frames: tuple[npt.NDArray[np.uint8], ...]) -> npt.NDArray[np.float32]:
        """Apply the checkpoint video processor to selected RGB frames."""
        if len(frames) != self._spec.video_frame_count:
            msg = f"Expected {self._spec.video_frame_count} selected RGB frames, got {len(frames)}"
            raise ValueError(msg)
        pixels = np.transpose(np.stack(frames)[None, ...], (0, 1, 4, 2, 3))
        videos = self._processor(videos=pixels)["videos"]
        prepared = videos[0].float().numpy()
        if prepared.shape != (
            self._spec.video_frame_count,
            3,
            self._spec.resolution,
            self._spec.resolution,
        ):
            msg = "Pinned processor returned an unexpected video tensor"
            raise ValueError(msg)
        return cast("npt.NDArray[np.float32]", prepared)


def load_video_preprocessor(model: VideoEmbeddingModelSpec, *, cpus: int = 1) -> VideoPreprocessor:
    """Load the CPU processor for a registered Cosmos-Embed1 variant."""
    return CosmosEmbed1VideoPreprocessor(model, cpus=cpus)


class CosmosEmbed1VideoEncoder:
    """Pinned Cosmos-Embed1 GPU model, loaded once per actor."""

    def __init__(self, model: VideoEmbeddingModelSpec) -> None:
        """Load the registered local checkpoint in the legacy-transformers GPU env."""
        import torch  # noqa: PLC0415 -- unavailable in the recipe's lightweight import path
        from transformers import AutoModel  # noqa: PLC0415 -- model runtime only

        if not torch.cuda.is_available():
            msg = "Cosmos-Embed1 video inference requires a visible CUDA GPU"
            raise RuntimeError(msg)
        model_dir = model.runtime_model_dir
        if not model_dir.is_dir():
            msg = f"Cosmos-Embed1 model weights are missing at {model_dir}"
            raise FileNotFoundError(msg)
        self._torch = torch
        self._model: Any = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True).to(
            "cuda", dtype=torch.bfloat16
        )
        self._model.eval()
        if self._model.config.embed_dim != model.vector_dimension:
            msg = "Cosmos-Embed1 model dimension differs from the registered contract"
            raise ValueError(msg)
        self._spec = model

    def encode_batch(self, videos: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run one batched model call; vector validation belongs to each result row."""
        if videos.ndim != _VIDEO_BATCH_NDIM or videos.shape[1] != self._spec.video_frame_count:
            msg = "Cosmos-Embed1 prepared video batch has an incompatible shape"
            raise ValueError(msg)
        tensor = self._torch.from_numpy(videos).to("cuda", dtype=self._torch.bfloat16)
        with self._torch.inference_mode():
            output = self._model.get_video_embeddings(videos=tensor)
        return cast("npt.NDArray[np.float32]", output.visual_proj.to("cpu", dtype=self._torch.float32).numpy())


def load_video_encoder(model: VideoEmbeddingModelSpec) -> VideoEncoder:
    """Load the GPU encoder for a registered Cosmos-Embed1 variant."""
    return CosmosEmbed1VideoEncoder(model)
