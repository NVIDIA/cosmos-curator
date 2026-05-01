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

"""Model Clip Asthetics."""

import numpy as np
import numpy.typing as npt
import torch

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.models.aesthetics import _AESTHETICS_MODEL_ID, AestheticScorer
from cosmos_curator.models.clip import _CLIP_MODEL_ID, CLIPImageEmbeddings


class CLIPAestheticScorer(ModelInterface):
    """A model that chains CLIPImageEmbeddings and AestheticScorer models."""

    def __init__(self) -> None:
        """Initialize the CLIPAestheticScorer model."""
        super().__init__()
        self._clip_model: CLIPImageEmbeddings | None = None
        self._aesthetic_model: AestheticScorer | None = None

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
        return [_AESTHETICS_MODEL_ID, _CLIP_MODEL_ID]

    def setup(self) -> None:
        """Set up the CLIPAestheticScorer model."""
        self._clip_model = CLIPImageEmbeddings()
        self._aesthetic_model = AestheticScorer()
        self._clip_model.setup()
        self._aesthetic_model.setup()

    def __call__(self, images: torch.Tensor | npt.NDArray[np.uint8]) -> torch.Tensor:
        """Call the CLIPAestheticScorer model.

        Args:
            images: The images to score.

        Returns:
            The scores.

        """
        assert self._clip_model
        assert self._aesthetic_model
        embeddings = self._clip_model(images)
        return self._aesthetic_model(embeddings)
