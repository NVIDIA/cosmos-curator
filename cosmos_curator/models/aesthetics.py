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

"""Model Aesthetics."""

import numpy as np
import numpy.typing as npt
import torch
from safetensors.torch import load_file
from torch import nn

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import model_utils

_AESTHETICS_MODEL_ID = "ttj/sac-logos-ava1-l14-linearMSE"


class MLP(nn.Module):
    """Multi-layer perceptron.

    A neural network that processes embeddings to predict aesthetic scores.
    """

    def __init__(self) -> None:
        """Initialize the MLP.

        Args:
            None

        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            embed: Input embeddings tensor.

        Returns:
            Predicted aesthetic scores.

        """
        return self.layers(embed)  # type: ignore[no-any-return]


class _AestheticScorer(torch.nn.Module):
    """Internal aesthetic scoring model implementation.

    This class handles the core aesthetic scoring functionality using a pre-trained MLP.
    """

    def __init__(self, weights_path: str) -> None:
        """Initialize the aesthetic scorer.

        Args:
            weights_path: Path to the model weights file.

        """
        super().__init__()
        self.weights_path = weights_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mlp = MLP()

        state_dict = load_file(weights_path)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.dtype = torch.float32
        self.eval()

    @torch.no_grad()
    def __call__(self, embeddings: torch.Tensor | npt.NDArray[np.float32]) -> torch.Tensor:
        """Score the aesthetics of input embeddings.

        Args:
            embeddings: Input embeddings as either a torch tensor or numpy array.

        Returns:
            Aesthetic scores for each input embedding.

        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings.copy())
        return self.mlp(embeddings.to(self.device)).squeeze(1)  # type: ignore[no-any-return]


class AestheticScorer(ModelInterface):
    """Public interface for aesthetic scoring of video embeddings.

    This class provides a standardized interface for scoring the aesthetic quality
    of video embeddings using a pre-trained model.
    """

    def __init__(self) -> None:
        """Initialize the aesthetic scorer interface."""
        super().__init__()

    @property
    def conda_env_name(self) -> str:
        """Get the name of the conda environment required for this model.

        Returns:
            Name of the conda environment.

        """
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names associated with this aesthetic scorer.

        Returns:
            A list containing the model ID for aesthetics scoring.

        """
        return [_AESTHETICS_MODEL_ID]

    def setup(self) -> None:
        """Set up the aesthetic scoring model by loading weights."""
        model_dir = model_utils.get_local_dir_for_weights_name(self.model_id_names[0])
        self._model = _AestheticScorer((model_dir / "model.safetensors").as_posix())

    def __call__(self, embeddings: torch.Tensor | npt.NDArray[np.float32]) -> torch.Tensor:
        """Score the aesthetics of input embeddings.

        Args:
            embeddings: Input embeddings as either a torch tensor or numpy array.

        Returns:
            Aesthetic scores for each input embedding.

        """
        return self._model(embeddings)
