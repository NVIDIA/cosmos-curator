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

"""Model Clips."""

from typing import Final

import numpy as np
import numpy.typing as npt
import torch

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import model_utils, pixi_utils

if pixi_utils.is_running_in_env("unified"):
    from torchvision import transforms  # type: ignore[import-untyped]
    from transformers import CLIPModel
else:
    transforms = None

_CLIP_MODEL_ID: Final = "openai/clip-vit-large-patch14"


class _CLIPImageEmbeddings(torch.nn.Module):
    def __init__(self, weights_name: str) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_file = model_utils.get_local_dir_for_weights_name(weights_name)
        self.clip = CLIPModel.from_pretrained(weight_file).to(self.device).eval()
        self.dtype = torch.float32

        # torchvision transforms that match CLIP preprocessor_config.json:
        if transforms is None:
            msg = "torchvision.transforms is unavailable; ensure you're in the 'unified' environment"
            raise RuntimeError(msg)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float32),  # scales [0, 255] to [0, 1]
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor | npt.NDArray[np.uint8]) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            # (N, H, W, C) -> (N, C, H, W)
            images = torch.from_numpy(images).permute(0, 3, 1, 2).to(self.device)

        inputs = self.transforms(images)
        embed = self.clip.get_image_features(pixel_values=inputs)

        # Normalize embeddings
        return embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)  # type: ignore[no-any-return]


class CLIPImageEmbeddings(ModelInterface):
    """Interface for generating CLIP image embeddings from input images."""

    def __init__(self) -> None:
        """Initialize the CLIPImageEmbeddings model."""
        super().__init__()

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model IDs used by this model.

        """
        return [_CLIP_MODEL_ID]

    def setup(self) -> None:
        """Set up the CLIPImageEmbeddings model."""
        self._model = _CLIPImageEmbeddings(self.model_id_names[0])

    def __call__(self, images: torch.Tensor | npt.NDArray[np.uint8]) -> torch.Tensor:
        """Call the CLIPImageEmbeddings model.

        Args:
            images: The images to embed.

        Returns:
            The embeddings.

        """
        return self._model(images)
